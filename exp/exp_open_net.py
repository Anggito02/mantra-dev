from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, AutoformerS1, Bautoformer, B2autoformer, B3autoformer, B4autoformer, B5autoformer, B6autoformer, B7autoformer, iTransformer, B6iFast, S1iSlow 
from models import Uautoformer, UautoformerC1, UautoformerC2, Uautoformer2, Transformer, Reformer, Mantra, MantraV1, MantraA, MantraB, MantraD, MantraE
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, NegativeCorr
from utils.slowloss import SlowLearnerLoss, ssl_loss, ssl_loss_v2

from utils.tools import visual

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os
import time

import warnings
import numpy as np

from tqdm import trange
from scipy.special import softmax
from sktime.performance_metrics.forecasting import \
    mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

warnings.filterwarnings('ignore')

class Exp_Main_DualmodE3K(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_DualmodE3K, self).__init__(args)

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
        self.slow_model = model_dict[self.args.slow_model].Model(self.args).float().to(self.device)
        # self.slow_model = model_dict['Autoformer'].Model(self.args).float().cuda()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # slow_model_optim = optim.Adam(self.slow_model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_slow_optimizer(self):
        slow_model_optim = optim.Adam(self.slow_model.parameters(), lr=self.args.learning_rate)
        return slow_model_optim

    def _select_criterion(self):
        criterion = NegativeCorr(self.args.corr_penalty) if self.args.loss == "neg_corr" else nn.MSELoss()
        return criterion

    # def _acquire_device(self):
    #     if self.args.use_multi_gpu:
    #         print("USE Multiple GPU")
    #     elif self.args.use_gpu:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(
    #             self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
    #         device = torch.device('cuda:{}'.format(self.args.gpu))
    #         print('Use GPU: cuda:{}'.format(self.args.gpu))
    #     else:
    #         device = torch.device('cpu')
    #         print('Use CPU')
    #     return device


    def vali(self, vali_data, vali_loader, criterion, setting, flag):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                f_dim = -1 if self.args.features == 'MS' else 0

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            # Outputs for every models
                            outputs, attns, arr_outputs, arr_attns = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            # Outputs for every models
                            outputs, arr_outputs = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        # Outputs for every models
                        outputs, attns, arr_outputs, arr_attns = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        # Outputs for every models
                        outputs, arr_outputs = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # model path
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # train result path
        result_path = self.args.checkpoints + setting + '/train_results/mantra/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        slow_model_optim = self._select_slow_optimizer()
        criterion = self._select_criterion()

        last_params = self.model.state_dict()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                f_dim = -1 if self.args.features == 'MS' else 0

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)


                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # print("enter with cuda amp")
                        if self.args.output_attention:
                            # Outputs for every models
                            outputs, attns, arr_outputs, arr_attns = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        else:
                            # Outputs for every models
                            outputs, arr_outputs = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    # print("enter without cuda amp")
                    if self.args.output_attention:
                        # Outputs for every models
                        outputs, attns, arr_outputs, arr_attns = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        # Outputs for every models
                        outputs, arr_outputs = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                need_update = True
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                    if (loss.item() > self.args.anomaly):
                        need_update = False
                        if i > 0:
                            self.model.load_state_dict(last_params)
                            clr = model_optim.param_groups[0]['lr']
                            model_optim = self._select_optimizer()
                            model_optim.param_groups[0]['lr'] = clr
                        else:
                            self.model = self._build_model()
                            clr = model_optim.param_groups[0]['lr']
                            model_optim = self._select_optimizer()
                            slow_model_optim = self._select_slow_optimizer()
                            model_optim.param_groups[0]['lr'] = clr
                            slow_model_optim.param_groups[0]['lr'] = clr

                    else:
                        last_params = self.model.state_dict()

                if(need_update):
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        # scaler.step(slow_model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                        # slow_model_optim.step()

                # introduce randomness for slow model batch_x
                # purpose:
                #   1. data augmentation (data training tambahan untuk training): add noise
                #   2. regularization: menghindari overfitting (tidak terlalu bergantung pada salah satu elemen)
                #   3. simulating noisy condition: digunakan agar model tetap bekerja dengan baik, meskipun ada noise
                if (self.args.use_slow_learner):
                    loss = 0
                    s0,s1,s2 = batch_x.shape
                    randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                    m_ones = torch.ones(s0,s1,s2).cuda()
                    slow_mark = torch.bernoulli(randuniform).cuda()
                    batch_x_slow = torch.clone(batch_x)
                    batch_x_slow = batch_x_slow * (m_ones-slow_mark)
                    
                    # decoder input
                    if self.args.seq_len < self.args.pred_len:
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.seq_len:, :]).float()
                    else:
                        dec_inp = torch.zeros_like(batch_x).float()
                        # batch_y_mark = torch.cat((batch_y_mark,batch_y_mark),dim=1)
                        d0,d1,d2 = batch_y_mark.shape
                        # d1=self.args.seq_len * 2
                        batch_y_mark2 = torch.empty(d0,d1,d2).uniform_(-1, 1).float().cuda()
                        # print(batch_y_mark)
                        batch_y_mark = torch.cat((batch_y_mark,batch_y_mark2),dim=1)

                    # dec_inp = torch.zeros_like(batch_y[:, -self.args.seq_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # forward slow model
                    if self.args.output_attention:
                        slow_out = self.slow_model.forward(batch_x_slow, batch_x_mark, dec_inp, batch_y_mark[:,:dec_inp.shape[1],:])[0]
                    else:
                        slow_out = self.slow_model.forward(batch_x_slow, batch_x_mark, dec_inp, batch_y_mark[:,:dec_inp.shape[1],:])


                    # compute slow loss
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = slow_out[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss += ssl_loss_v2(slow_out, batch_x, slow_mark, s1, s2)

                    if(need_update):
                        slow_model_optim.zero_grad()
                        if self.args.use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(slow_model_optim)
                            scaler.step(model_optim)
                            scaler.update()
                        else:
                            loss.backward()
                            slow_model_optim.step()
                            model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, setting, "vali")
            test_loss = self.vali(test_data, test_loader, criterion, setting, "test")

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)

            with open(f'{result_path}/mantra_log.txt', 'a') as f:
                f.write(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}\n\n".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './checkpoints/' + setting + '/testing_results/mantra/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        isFirst = True
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, attns, _, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, attns, _, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    else:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                if isFirst:
                    isFirst = False
                    preds = np.array(pred)
                    trues = np.array(true)

                else:
                    preds = np.concatenate((preds,pred), axis=0)
                    trues = np.concatenate((trues,true), axis=0)
                
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '_with_gt.pdf'))

        
        # for i in range(0,90):
        #     print(preds[i].shape)

        # for i in range(0,90):
        #     print(preds[i].shape)

        # preds = np.array(preds)
        # trues = np.array(trues)
        # preds = np.stack(preds,axis=0)
        # trues = np.stack(trues,axis=0)

        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # preds = np.array(preds.flat)
        # trues = np.array(trues.flat)
        # print(preds[0].shape)
        # print(trues[0].shape)

        # # print(trues)
        # print(preds.shape)
        # print(trues.shape)
        # print('test shape:', preds.shape, trues.shape)

        # result save
        result_path = './checkpoints/' + setting + '/testing_results/'

        print('test shape:', preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        with open(result_path + 'result_mantra.txt', 'a') as f:
            f.write(setting + '  \n')
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')

        with open(folder_path + 'metrics.npy', 'wb') as f:
            np.save(f, np.array([mae, mse, rmse, mape, mspe]))
        with open(folder_path + 'pred.npy', 'wb') as f:
            np.save(f, preds)
        with open(folder_path + 'true.npy', 'wb') as f:
            np.save(f, trues)

        # for i in range(0, self.args.n_learner):
        #     print("Test learner: "+str(i)+" ", end="")
        #     self.test_1learner(setting, test, i)
        return

    def set_rl_data(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        model_path = os.path.join(self.args.checkpoints, setting)
        # Load the model
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        assert os.path.exists(model_path), "cannot find {} model path".format(model_path)

        print("Set training data...\n")
        train_inputs, train_preds, train_trues = self._forward_rl(train_data, train_loader, 10)
        print("Set validation data...\n")
        vali_inputs, vali_preds, vali_trues = self._forward_rl(vali_data, vali_loader, 10)
        print("Set testing data...\n")
        test_inputs, test_preds, test_trues = self._forward_rl(test_data, test_loader)

        bm_train_preds = {}
        bm_vali_preds = {}
        bm_test_preds = {}

        for model_idx in range(self.args.n_learner):
            bm_train_preds['Learner_' + str(model_idx)] = train_preds[model_idx]
            bm_vali_preds['Learner_' + str(model_idx)] = vali_preds[model_idx]
            bm_test_preds['Learner_' + str(model_idx)] = test_preds[model_idx]

        # save rl data
        if not os.path.exists(f'{model_path}/dataset'):
            os.makedirs(f'{model_path}/dataset')

        if not os.path.exists(f'{model_path}/rl_bm'):
            os.makedirs(f'{model_path}/rl_bm')

        with open(f'{model_path}/dataset/input_train_x.npy', 'wb') as f:
            np.save(f, train_inputs)
        with open(f'{model_path}/dataset/input_train_y.npy', 'wb') as f:
            np.save(f, train_trues)
        with open(f'{model_path}/dataset/input_vali_x.npy', 'wb') as f:
            np.save(f, vali_inputs)
        with open(f'{model_path}/dataset/input_vali_y.npy', 'wb') as f:
            np.save(f, vali_trues)
        with open(f'{model_path}/dataset/input_test_x.npy', 'wb') as f:
            np.save(f, test_inputs)
        with open(f'{model_path}/dataset/input_test_y.npy', 'wb') as f:
            np.save(f, test_trues)

        with open(f'{model_path}/rl_bm/bm_train_preds.npz', 'wb') as f:
            np.savez(f, **bm_train_preds)
        with open(f'{model_path}/rl_bm/bm_vali_preds.npz', 'wb') as f:
            np.savez(f, **bm_vali_preds)
        with open(f'{model_path}/rl_bm/bm_test_preds.npz', 'wb') as f:
            np.savez(f, **bm_test_preds)

        return
    
    def _forward_rl(self, flag_data, flag_loader, epoch=1):
        inputs = []
        trues = []
        preds = []

        self.model.eval()
        with torch.no_grad():
            for _ in trange(epoch, desc='[Get Data Epoch]'):
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(flag_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs, attns, arr_outputs, arr_attns = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            else:
                                outputs, arr_outputs = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs, attns, arr_outputs, arr_attns = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        else:
                            outputs, arr_outputs = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0

                    for i in range(self.args.n_learner):
                        arr_outputs[i] = arr_outputs[i][:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                    batch_x = batch_x.detach().cpu().numpy()
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    batch_y = batch_y.detach().cpu().numpy()

                    arr_outputs = np.array(arr_outputs)
                    input = batch_x
                    true = batch_y

                    preds.append(arr_outputs)
                    inputs.append(input)
                    trues.append(true)

            preds = np.concatenate(preds, axis=1)
            inputs = np.concatenate(inputs)
            trues = np.concatenate(trues)

            return inputs, preds, trues
    

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
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
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return