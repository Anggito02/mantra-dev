import os
import torch
import torch.nn as nn
import gymnasium as gym

from torch import optim
from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.slowloss import ssl_loss_v2
import numpy as np
from tqdm import trange
from exp.exp_rl import Env

from models import Informer, Autoformer, AutoformerS1, Bautoformer, B2autoformer, B3autoformer, B4autoformer, B5autoformer, B6autoformer, B7autoformer, iTransformer, B6iFast, S1iSlow 
from models import Uautoformer, UautoformerC1, UautoformerC2, Uautoformer2, Transformer, Reformer, Mantra, MantraV1, MantraA, MantraB, MantraD, MantraE

from stable_baselines3 import DDPG

from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, NegativeCorr
from utils.slowloss import SlowLearnerLoss, ssl_loss, ssl_loss_v2

class OPT_RL_OpenNet(Exp_Basic):
    def __init__(self, args):
        super(OPT_RL_OpenNet, self).__init__(args)
        self.model = None
        self.device = self._acquire_device()

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'AutoformerS1': AutoformerS1,
            'Bautoformer': Bautoformer,
            'B2autoformer': B2autoformer,
            'B3autoformer': B3autoformer,
            'B4autoformer': B4autoformer,
            'B5autoformer': B5autoformer,
            'B6autoformer': B6autoformer,
            'B7autoformer': B7autoformer,
            'Mantra': Mantra,
            'MantraV1': MantraV1,
            'MantraA': MantraA,
            'MantraB': MantraB,
            'MantraD': MantraD,
            'MantraE': MantraE,
            'Uautoformer': Uautoformer,
            'UautoformerC1': UautoformerC1,
            'UautoformerC2': UautoformerC2,
            'Uautoformer2': Uautoformer2,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'iTransformer' : iTransformer,
            'B6iFast' : B6iFast,
            'S1iSlow' : S1iSlow,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_slow_optimizer(self):
        slow_model_optim = optim.Adam(self.slow_model.parameters(), lr=self.args.learning_rate)
        return slow_model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def valid_model(self, setting, valid_data, valid_loader):
        pass

    def test_model(self, setting, flag_data, flag_loader):
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(flag_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if flag_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = flag_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = flag_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

            preds = np.array(preds)
            trues = np.array(trues)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            return mse, mae, mape, preds, trues
    
    def opt_rl_train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        model_path = os.path.join(self.args.checkpoints, setting)
        assert os.path.exists(model_path), "cannot find {} model path".format(model_path)
        model_list = [model_state for model_state in os.listdir(model_path) if model_state.endswith(".pth")]

        train_model_errors = []
        valid_model_errors = []
        test_model_errors = []

        for model_idx in trange(len(model_list), desc='[Optimize RL]'):
            model_name = model_list[model_idx].split(".")[0].split("_")[-1]

            # Load the model
            self.model = self.model_dict[model_name].Model(self.args).float().to(self.device)
            model_load_state = torch.load(os.path.join("./checkpoints/", setting, model_list[model_idx]))
            self.model.load_state_dict(model_load_state)
            
            # Get Train data
            preds, trues, mse, mae, mape = self.test_model(setting, train_data, train_loader)
            train_model_errors.append([preds, trues, mse, mae, mape])

            # Get Vali data
            preds, trues, mse, mae, mape = self.test_model(setting, vali_data, vali_loader)
            valid_model_errors.append([preds, trues, mse, mae, mape])

            # Get Test data
            preds, trues, mse, mae, mape = self.test_model(setting, test_data, test_loader)
            test_model_errors.append([preds, trues, mse, mae, mape])


        model_train_mse_loss = []
        model_train_gt = []
        model_train_preds = 

        for i in range(len(train_model_errors)):
            model_train_mse_loss.append(train_model_errors[i][2])
            model_train_gt.append(test_model_errors[i][1])
        
        env = Env(model_train_mse_loss, model_train_gt, )




