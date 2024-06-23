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
from utils.open_net_env import OpenNetEnv, evaluate_policy

from models import Informer, Autoformer, AutoformerS1, Bautoformer, B2autoformer, B3autoformer, B4autoformer, B5autoformer, B6autoformer, B7autoformer, iTransformer, B6iFast, S1iSlow 
from models import Uautoformer, UautoformerC1, UautoformerC2, Uautoformer2, Transformer, Reformer, Mantra, MantraV1, MantraA, MantraB, MantraD, MantraE

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

class OPT_RL_OpenNet(Exp_Basic):
    def __init__(self, args):
        super(OPT_RL_OpenNet, self).__init__(args)
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

    def test_model(self, flag_data, flag_loader):
        sources = []
        trues = []
        preds = []

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
                batch_x = batch_x.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                source = batch_x
                pred = outputs
                true = batch_y

                sources.append(source)
                preds.append(pred)
                trues.append(true)

            sources = np.concatenate(sources)
            preds = np.concatenate(preds)
            trues = np.concatenate(trues)

            return sources, preds, trues
    
    def opt_rl_train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        model_path = os.path.join(self.args.checkpoints, setting)
        assert os.path.exists(model_path), "cannot find {} model path".format(model_path)
        model_list = [model_state for model_state in os.listdir(model_path) if model_state.endswith(".pth")]

        list_train_X = []
        list_train_Y = []
        list_valid_X = []
        list_valid_Y = []
        list_test_X = []
        list_test_Y = []
        train_preds = []
        vali_preds = []
        test_preds = []

        for model_idx in trange(len(model_list), desc='[Set RL Data]'):
            model_name = model_list[model_idx].split(".")[0].split("_")[-1]

            # Load the model
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, f'checkpoint_fast_{self.args.model_idx}_{self.args.model}.pth')))
            
            # Get Train data
            sources, preds, trues = self.test_model(train_data, train_loader)

            list_train_X.append(sources)
            list_train_Y.append(preds)
            train_preds.append(trues)

            # Get Validation data
            sources, preds, trues = self.test_model(vali_data, vali_loader)

            list_valid_X.append(sources)
            list_valid_Y.append(preds)
            vali_preds.append(trues)

            # Get Test data
            sources, preds, trues = self.test_model(test_data, test_loader)

            list_test_X.append(sources)
            list_test_Y.append(preds)
            test_preds.append(trues)

        list_train_X = np.concatenate(list_train_X)
        list_train_Y = np.concatenate(list_train_Y)
        list_valid_X = np.concatenate(list_valid_X)
        list_valid_Y = np.concatenate(list_valid_Y)
        list_test_X = np.concatenate(list_test_X)
        list_test_Y = np.concatenate(list_test_Y)
            
        bm_train_preds = {}
        bm_vali_preds = {}
        bm_test_preds = {}

        for model_idx in range(len(model_list)):
            bm_train_preds['Learner_' + str(model_idx)] = train_preds[model_idx]
            bm_vali_preds['Learner_' + str(model_idx)] = vali_preds[model_idx]
            bm_test_preds['Learner_' + str(model_idx)] = test_preds[model_idx]

        # save rl data
        if not os.path.exists(f'{model_path}/dataset'):
            os.makedirs(f'{model_path}/dataset')

        if not os.path.exists(f'{model_path}/rl_bm'):
            os.makedirs(f'{model_path}/rl_bm')

        with open(f'{model_path}/dataset/input_train_x.npy', 'wb') as f:
            np.save(f, list_train_X)
        with open(f'{model_path}/dataset/input_train_y.npy', 'wb') as f:
            np.save(f, list_train_Y)
        with open(f'{model_path}/dataset/input_vali_x.npy', 'wb') as f:
            np.save(f, list_valid_X)
        with open(f'{model_path}/dataset/input_vali_y.npy', 'wb') as f:
            np.save(f, list_valid_Y)
        with open(f'{model_path}/dataset/input_test_x.npy', 'wb') as f:
            np.save(f, list_test_X)
        with open(f'{model_path}/dataset/input_test_y.npy', 'wb') as f:
            np.save(f, list_test_Y)

        with open(f'{model_path}/rl_bm/bm_train_preds.npz', 'wb') as f:
            np.save(f, **bm_train_preds)
        with open(f'{model_path}/rl_bm/bm_vali_preds.npz', 'wb') as f:
            np.save(f, **bm_vali_preds)
        with open(f'{model_path}/rl_bm/bm_test_preds.npz', 'wb') as f:
            np.save(f, **bm_test_preds)

        # RL training
        train_env = OpenNetEnv(model_path, len(model_list), mode="train")

        n_actions = len(model_list)
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = DDPG("MlpPolicy", train_env, action_noise=action_noise, verbose=1)

        model.learn(total_timesteps=list_train_X.shape[0])

        # RL validation
        vali_env = OpenNetEnv(model_path, len(model_list), mode="vali")
        mse_loss, mae_loss, mean_reward, std_reward = evaluate_policy(model, vali_env, n_eval_episodes=1)
        print(f"\nRL Validation\nmean_reward={mean_reward:.2f} +/- {std_reward:.2f}\nMSE={mse_loss:.3f}, MAE={mae_loss:.3f}")

        model.save(f"{model_path}/rl_model")

        del model, train_env, vali_env

        # RL testing
        model = DDPG.load(f"{model_path}/rl_model")

        test_env = OpenNetEnv(model_path, len(model_list), mode="test")
        mse_loss, mae_loss, mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=1)
        print(f"\nRL Test\nmean_reward={mean_reward:.2f} +/- {std_reward:.2f}\nMSE={mse_loss:.3f}, MAE={mae_loss:.3f}")
