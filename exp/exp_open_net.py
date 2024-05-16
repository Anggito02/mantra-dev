from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, AutoformerS1, Bautoformer, B2autoformer, B3autoformer, B4autoformer, B5autoformer, B6autoformer, B7autoformer, iTransformer, B6iFast, S1iSlow 
from models import Uautoformer, UautoformerC1, UautoformerC2, Uautoformer2, Transformer, Reformer, Mantra, MantraV1, MantraA, MantraB, MantraD, MantraE
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, NegativeCorr
from utils.slowloss import SlowLearnerLoss, ssl_loss, ssl_loss_v2
from utils.utils import unify_input_data, load_data, get_state_weight, get_batch_reward, sparse_explore, evaluate_agent
from utils.tools import visual

from models.ddpg import Actor, Critic

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os
import time
from tqdm import trange

import warnings
import matplotlib.pyplot as plt
import numpy as np
import random

from collections import Counter
from scipy.special import softmax
from sktime.performance_metrics.forecasting import \
    mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

warnings.filterwarnings('ignore')

class DDPGAgent:
    def __init__(self, device, use_td, states, obs_dim, act_dim, hidden_dim=256,
                 lr=3e-4, gamma=0.99, tau=0.005):
        # initialize the actor & target_actor
        self.actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
        self.target_actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # initialize the critic
        self.critic = Critic(obs_dim, act_dim, hidden_dim).to(device)
        self.target_critic = Critic(obs_dim, act_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # training states
        self.states = states

        # parameters
        self.gamma  = gamma
        self.tau    = tau
        self.use_td = use_td

        # update the target network
        for param, target_param in zip(
                self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(param.data)
        for param, target_param in zip(
                self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(param.data)

    def select_action(self, obs):
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
        return softmax(action, axis=1)

    def update(self,
               sampled_obs_idxes,
               sampled_actions,
               sampled_rewards,
               sampled_weights=None):
        batch_obs = self.states[sampled_obs_idxes]  # (512, 7, 20)

        with torch.no_grad():
            if self.use_td:
                # update w.r.t the TD target
                batch_next_obs = self.states[sampled_obs_idxes + 1]
                target_q = self.target_critic(
                    batch_next_obs, self.target_actor(batch_next_obs))  # (B,)
                target_q = sampled_rewards + self.gamma * target_q  # (B,)
            else:
                # without TD learning, just is supervised learning
                target_q = sampled_rewards
        current_q = self.critic(batch_obs, sampled_actions)     # (B,)

        # critic loss
        if sampled_weights is None:
            q_loss = F.mse_loss(current_q, target_q)
        else:
            # weighted mse loss
            q_loss = (sampled_weights * (current_q - target_q)**2).sum() /\
                sampled_weights.sum()

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # actor loss ==> convert actor output to softmax weights
        if sampled_weights is None:
            actor_loss = -self.critic(
                batch_obs, F.softmax(self.actor(batch_obs), dim=1)).mean()
        else:
            # weighted actor loss
            actor_loss = -self.critic(batch_obs, F.softmax(self.actor(batch_obs), dim=1))
            actor_loss = (sampled_weights * actor_loss).sum() / sampled_weights.sum()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        if self.use_td:
            for param, target_param in zip(
                    self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(
                self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q_loss': q_loss.item(),
            'pi_loss': actor_loss.item(),
            'current_q': current_q.mean().item(),
            'target_q': target_q.mean().item()
        }


class Env:
    def __init__(self, train_error, train_y, bm_train_preds):
        self.error = train_error
        self.bm_preds = bm_train_preds
        self.y = train_y
    
    def reward_func(self, idx, action):
        if isinstance(action, int):
            tmp = np.zeros(self.bm_preds.shape[1])
            tmp[action] = 1.
            action = tmp
        weighted_y = np.multiply(action.reshape(-1, 1), self.bm_preds[idx])
        weighted_y = weighted_y.sum(axis=0)
        new_mape = mean_absolute_percentage_error(self.y[idx], weighted_y)
        new_mae = mean_absolute_error(self.y[idx], weighted_y)
        new_mse = mean_squared_error(self.y[idx], weighted_y)
        new_error = np.array([*self.error[idx], new_mse])
        rank = np.where(np.argsort(new_error) == len(new_error) - 1)[0][0]
        return rank, new_mape, new_mae, new_mse


class ReplayBuffer:
    def __init__(self, device, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        # In TS data, `next_state` is just the S[i+1]
        self.states = np.zeros((max_size, 1), dtype=np.int32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=256):
        ind = np.random.randint(self.size, size=batch_size)
        states = self.states[ind].squeeze()
        actions = torch.FloatTensor(self.actions[ind]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[ind]).to(self.device)
        return (states, actions, rewards.squeeze())
    
def pretrain_actor(device, obs_dim, act_dim, hidden_dim, states, train_error, cls_weights, 
                   valid_states, valid_error):
    best_train_model = torch.LongTensor(train_error.argmin(1)).to(device)
    best_valid_model = torch.LongTensor(valid_error.argmin(1)).to(device)

    actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
    best_actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
    cls_weights = torch.FloatTensor([1/cls_weights[w] for w in range(act_dim)]).to(device)

    L = len(states)
    batch_size = 512
    batch_num  = int(np.ceil(L / batch_size))
    optimizer  = torch.optim.Adam(actor.parameters(), lr=3e-4)
    loss_fn    = nn.CrossEntropyLoss(weight=cls_weights)  # weighted CE loss
    best_acc   = 0
    patience   = 0
    max_patience = 5
    for epoch in trange(200, desc='[Pretrain]'):
        epoch_loss = []
        shuffle_idx = np.random.permutation(np.arange(L))
        for i in range(batch_num):
            batch_idx = shuffle_idx[i*batch_size: (i+1)*batch_size]
            optimizer.zero_grad()
            batch_out = actor(states[batch_idx])
            loss = loss_fn(batch_out, best_train_model[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        with torch.no_grad():
            pred = actor(valid_states)
            pred_idx = pred.argmax(1)
            acc = (pred_idx == best_valid_model).sum() / len(pred)
        print(f'# epoch {epoch+1}: loss = {np.average(epoch_loss):.5f}\tacc = {acc:.3f}')

        # early stop w.r.t. validation acc
        if acc > best_acc:
            best_acc = acc
            patience = 0
            # update best model
            for param, target_param in zip(
                    actor.parameters(), best_actor.parameters()):
                target_param.data.copy_(param.data)
        else:
            patience += 1
        
        if patience == max_patience:
            break

    with torch.no_grad():
        pred = best_actor(valid_states)
        pred_idx = pred.argmax(1)
        acc = (pred_idx == best_valid_model).sum() / len(pred)    
    print(f'valid acc for pretrained actor: {acc:.3f}') 
    return best_actor

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

            # bm train preds npz
            bm_flag_preds = [[] for _ in range(self.args.n_learner)]

            # flag data
            input_flag_x = []
            input_flag_y = []

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                f_dim = -1 if self.args.features == 'MS' else 0

                temp_x, temp_y = batch_x, batch_y
        
                input_flag_x.append(temp_x.numpy())

                input_flag_x.append(temp_x.numpy())
                input_flag_y.append(temp_y[:, -self.args.pred_len:, -1].numpy())

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
                            for models_idx in range(self.args.n_learner):
                                outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark, idx=models_idx)[0]

                                # Save bm_flag_preds
                                bm_flag_preds[models_idx].append(outputs[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy())

                            # For Mantra
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            # Outputs for every models
                            for models_idx in range(self.args.n_learner):
                                outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark, idx=models_idx)

                                # Save bm_flag_preds
                                bm_flag_preds[models_idx].append(outputs[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy())
                            
                            # For Mantra
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        # Outputs for every models
                        for models_idx in range(self.args.n_learner):
                            outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark, idx=models_idx)[0]

                            # Save bm_flag_preds
                            bm_flag_preds[models_idx].append(outputs[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy())

                        # For Mantra
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        # Outputs for every models
                        for models_idx in range(self.args.n_learner):
                            outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark, idx=models_idx)

                            # Save bm_flag_preds
                            bm_flag_preds[models_idx].append(outputs[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy())

                        # For Mantra
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

            # Save input vali and test
            np_input_flag_x = input_flag_x[0]
            np_input_flag_y = input_flag_y[0]

            for i in range(1, len(input_flag_x)):
                np_input_flag_x = np.append(np_input_flag_x, input_flag_x[i], axis=0)

            for i in range(1, len(input_flag_y)):
                np_input_flag_y = np.append(np_input_flag_y, input_flag_y[i], axis=0)    

            # np_input_flag_x = np.array(input_flag_x)
            # np_input_flag_x = np_input_flag_x.reshape(np_input_flag_x.shape[0]*np_input_flag_x.shape[1], np_input_flag_x.shape[2], np_input_flag_x.shape[3])

            # np_input_flag_y = np.array(input_flag_y)
            # np_input_flag_y = np_input_flag_y.reshape(np_input_flag_y.shape[0]*np_input_flag_y.shape[1], np_input_flag_y.shape[2], np_input_flag_y.shape[3])

            if not os.path.exists(self.args.checkpoints + '/' + setting + '/dataset/'):
                os.makedirs(self.args.checkpoints + '/' + setting + '/dataset/')

            with open(self.args.checkpoints + '/' + setting + '/dataset/' + f'input_{flag}_x.npy', 'wb') as f:
                np.save(f, np_input_flag_x)
            with open(self.args.checkpoints + '/' + setting + '/dataset/' + f'input_{flag}_y.npy', 'wb') as f:
                np.save(f, np_input_flag_y)

            # Update numpy bm_flag_preds
            bm_flag_preds_npz = {}
            for models_idx in range(self.args.n_learner):
                bm_flag_preds_npz['Learner_' + str(models_idx)] = np.concatenate(bm_flag_preds[models_idx], axis=0)

            bm_flag_preds_npz_path = self.args.checkpoints + '/' + setting + '/' + 'rl_bm' + '/'
            if not os.path.exists(bm_flag_preds_npz_path):
                os.makedirs(bm_flag_preds_npz_path)
            
            bm_flag_preds_npz_path = os.path.join(self.args.checkpoints, setting, 'rl_bm', f'bm_{flag}_preds.npz')
            with open(bm_flag_preds_npz_path, 'wb') as f:
                np.savez(f, **bm_flag_preds_npz)

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

        # bm train preds npz
        bm_train_preds = [[] for _ in range(self.args.n_learner)]

        # Initialize Input train, vali, test 3 dimension numpy array
        input_train_x = []
        input_train_y = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                f_dim = -1 if self.args.features == 'MS' else 0

                temp_x, temp_y = batch_x, batch_y
        
                input_train_x.append(temp_x.numpy())

                input_train_x.append(temp_x.numpy())
                input_train_y.append(temp_y[:, -self.args.pred_len:, -1].numpy())

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
                            for models_idx in range(self.args.n_learner):
                                outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark, idx=models_idx)[0]

                                # Save bm_train_preds
                                bm_train_preds[models_idx].append(outputs[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy())

                            # For Mantra
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            # Outputs for every models
                            for models_idx in range(self.args.n_learner):
                                outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark, idx=models_idx)

                                # Save bm_train_preds
                                bm_train_preds[models_idx].append(outputs[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy())

                            # For Mantra
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    # print("enter without cuda amp")
                    if self.args.output_attention:
                        # Outputs for every models
                        for models_idx in range(self.args.n_learner):
                            outputs = self.model.forward_1learner(batch_x, batch_x_mark,dec_inp, batch_y_mark, idx=models_idx)[0]

                            # Save bm_train_preds
                            bm_train_preds[models_idx].append(outputs[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy())
                        
                        # For Mantra
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        # Outputs for every models
                        for models_idx in range(self.args.n_learner):
                            outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark, idx=models_idx)

                            # Save bm_train_preds
                            bm_train_preds[models_idx].append(outputs[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy())
                        # For Mantra
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

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

                # FIXME: Implementasi Slow model
                # loss = 0
                # s0,s1,s2 = batch_x.shape
                # randuniform = torch.empty(s0,s1,s2).uniform_(0, 1)
                # m_ones = torch.ones(s0,s1,s2).cuda()
                # slow_mark = torch.bernoulli(randuniform).cuda()
                # batch_x_slow = torch.clone(batch_x)
                # batch_x_slow = batch_x_slow * (m_ones-slow_mark)
                
                #Update for slow model
                # loss = 0
                # s0,s1,s2 = batch_x.shape
                # slow_mark = torch.zeros(s0,s1,s2).cuda()
                # # slow_mark = torch.zeros(s1).cuda()
                # batch_x_slow = torch.clone(batch_x)
                # for c in range(0,self.args.n_learner):
                #     # idx = self.model.decoder[c].layers[self.args.d_layers-1].self_attention.inner_correlation.top_k_index
                #     idx = self.model.encoder[c].attn_layers[0].attention.inner_correlation.top_k_index
                #     # slow_mark[idx] = 1
                #     slow_mark[:,idx,:] = 1
                #     batch_x_slow[:,idx,:] = 0

                # loss = 0
                # s0,s1,s2 = batch_x.shape
                # slow_mark = torch.zeros(s0,s1,s2).cuda()
                # batch_x_slow = torch.clone(batch_x)
                # c = random.randint(0,self.args.n_learner-1)
                # idx = self.model.encoder[c].attn_layers[0].attention.inner_correlation.top_k_index
                # slow_mark[:,idx,:] = 1
                # batch_x_slow[:,idx,:] = 0
                
                # for s in range(0,s1):
                #     if slow_mark[s] == 1:
                #         batch_x_slow[:,s,:] = 0
                
                # minS1 = min(self.args.label_len, self.args.pred_len)
                # dec_inp = torch.zeros_like(batch_y[:, -minS1:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # if self.args.seq_len < self.args.pred_len:
                #     dec_inp = torch.zeros_like(batch_y[:, -self.args.seq_len:, :]).float()
                # else:
                #     dec_inp = torch.zeros_like(batch_x).float()
                #     # batch_y_mark = torch.cat((batch_y_mark,batch_y_mark),dim=1)
                #     d0,d1,d2 = batch_y_mark.shape
                #     # d1=self.args.seq_len * 2
                #     batch_y_mark2 = torch.empty(d0,d1,d2).uniform_(-1, 1).float().cuda()
                #     # print(batch_y_mark)
                #     batch_y_mark = torch.cat((batch_y_mark,batch_y_mark2),dim=1)

                # # dec_inp = torch.zeros_like(batch_y[:, -self.args.seq_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # if self.args.output_attention:
                #     # print("enter if")
                #     # fast_outputs = self.model.forward_for_slowlearner(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #     # slow_out = self.slow_model.forward_for_slowlearner(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #     # slow_out = self.slow_model.forward_for_slowlearner(batch_x_slow, batch_x_mark, dec_inp, batch_y_mark)[0]
                #     # slow_out = self.slow_model.forward_for_slowlearner(batch_x_slow, batch_x_mark, dec_inp, batch_y_mark[:,:dec_inp.shape[1],:])[0]
                #     slow_out = self.slow_model.forward(batch_x_slow, batch_x_mark, dec_inp, batch_y_mark[:,:dec_inp.shape[1],:])[0]
                    
                #     # # print("check fast and slow output:")
                #     # print(fast_outputs.shape)
                #     # print(slow_out.shape)
                #     # print(fast_outputs.shape)
                #     # for nl in range(0,fast_outputs.shape[0]):
                #     #     fast_output_nl = fast_outputs[nl]
                #     #     slow_out = self.slow_model.forward_and_modulate(batch_x, batch_x_mark, dec_inp, batch_y_mark,fast_output=fast_output_nl)
                #     #     # print(slow_out.shape)

                #     #     f_dim = -1 if self.args.features == 'MS' else 0
                #     #     outputs = slow_out[:, -self.args.pred_len:, f_dim:]
                #     #     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                #     #     loss += criterion(outputs, batch_y)


                #         # slow_model_optim.zero_grad()
                #         # if self.args.use_amp:
                #         #     scaler.scale(loss).backward()
                #         #     scaler.step(slow_model_optim)
                #         #     scaler.update()
                #         # else:
                #         #     loss.backward()
                #         #     slow_model_optim.step()

                # else:
                #     # print("enter else")
                #     # fast_outputs = self.model.forward_for_slowlearner(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                #     # slow_out = self.slow_model.forward_for_slowlearner(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                #     # slow_out = self.slow_model.forward_for_slowlearner(batch_x_slow, batch_x_mark, dec_inp, batch_y_mark)
                #     # slow_out = self.slow_model.forward_for_slowlearner(batch_x_slow, batch_x_mark, dec_inp, batch_y_mark[:,:dec_inp.shape[1],:])
                #     slow_out = self.slow_model.forward(batch_x_slow, batch_x_mark, dec_inp, batch_y_mark[:,:dec_inp.shape[1],:])
                #     # print("check fast and slow output:")
                #     # print(fast_outputs.shape)
                #     # print(slow_out.shape)
                #     # for nl in range(0,fast_outputs.shape[0]):
                #     #     fast_output_nl = fast_outputs[nl]
                #     #     slow_out = self.slow_model.forward_and_modulate(batch_x, batch_x_mark, dec_inp, batch_y_mark,fast_output=fast_output_nl)
                #     #     # print(slow_out.shape)


                # outputs = slow_out[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # loss += ssl_loss_v2(slow_out, batch_x, slow_mark, s1, s2)

                # # loss += criterion(outputs, batch_y)
                # # loss += ssl_loss(slow_out, batch_x_slow, slow_mark, s1, s2)
                # # loss += ssl_loss(slow_out, batch_x, slow_mark, s1, s2)
                # # loss += ssl_loss(slow_out, batch_x, slow_mark, s1, s2)
                # # loss += ssl_loss(slow_out, batch_x, batch_x_mark, s1, s2)
                

                # if(need_update):
                #     slow_model_optim.zero_grad()
                #     if self.args.use_amp:
                #         scaler.scale(loss).backward()
                #         scaler.step(slow_model_optim)
                #         scaler.step(model_optim)
                #         scaler.update()
                #     else:
                #         loss.backward()
                #         slow_model_optim.step()
                #         model_optim.step()

                # print(fast_outputs.shape)
                # print(outputs.shape)

            # print(">>>>>>> Check after epoch >>>>>>>")
            # self.model.check_params()

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

        # Save input train
        np_input_train_x = input_train_x[0]
        np_input_train_y = input_train_y[0]

        for i in range(1, len(input_train_x)):
            np_input_train_x = np.append(np_input_train_x, input_train_x[i], axis=0)
        
        for i in range(1, len(input_train_y)):
            np_input_train_y = np.append(np_input_train_y, input_train_y[i], axis=0)

        # np_input_train_x = np.array(input_train_x)
        # np_input_train_x = np_input_train_x.reshape(np_input_train_x.shape[0]*np_input_train_x.shape[1], np_input_train_x.shape[2], np_input_train_x.shape[3])

        # np_input_train_y = np.array(input_train_y)
        # np_input_train_y = np_input_train_y.reshape(np_input_train_y.shape[0]*np_input_train_y.shape[1], np_input_train_y.shape[2], np_input_train_y.shape[3])

        if not os.path.exists(self.args.checkpoints + '/' + setting + '/dataset/'):
            os.makedirs(self.args.checkpoints + '/' + setting + '/dataset/')

        with open(self.args.checkpoints + '/' + setting + '/dataset/'  'input_train_x.npy', 'wb') as f:
            np.save(f, np_input_train_x)
        with open(self.args.checkpoints + '/' + setting + '/dataset/' + 'input_train_y.npy', 'wb') as f:
            np.save(f, np_input_train_y)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # Update numpy bm_train_preds
        bm_train_preds_npz = {}
        for models_idx in range(self.args.n_learner):
            bm_train_preds_npz['Learner_' + str(models_idx)] = np.concatenate(bm_train_preds[models_idx], axis=0)

        bm_train_preds_npz_path = self.args.checkpoints + '/' + setting + '/' + 'rl_bm' + '/'
        if not os.path.exists(bm_train_preds_npz_path):
            os.makedirs(bm_train_preds_npz_path)
            
        bm_train_preds_npz_path = os.path.join(self.args.checkpoints, setting, 'rl_bm', 'bm_train_preds.npz')
        with open(bm_train_preds_npz_path, 'wb') as f:
            np.savez(f, **bm_train_preds_npz)

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
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

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
        preds_ot = preds[:, :, -1]
        trues_ot = trues[:, :, -1]

        for i in range(len(preds_ot)):
            visual(trues_ot[i], preds_ot[i], os.path.join(folder_path, str(i) + '.pdf'))

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



    def test_1learner(self, setting, test=0, idx=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
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
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            if self.args.use_multi_gpu and self.args.use_gpu:
                                outputs = self.model.module.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)[0]
                            else:
                                outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)[0]
                        else:
                            if self.args.use_multi_gpu and self.args.use_gpu:
                                outputs = self.model.module.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)
                            else:
                                outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)
                else:
                    if self.args.output_attention:
                        if self.args.use_multi_gpu and self.args.use_gpu:
                            outputs = self.model.module.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)[0]
                        else:
                            outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)[0]
                    else:
                        if self.args.use_multi_gpu and self.args.use_gpu:
                            outputs = self.model.module.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)
                        else:
                            outputs = self.model.forward_1learner(batch_x, batch_x_mark, dec_inp, batch_y_mark,idx=idx)
                            

                f_dim = -1 if self.args.features == 'MS' else 0
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
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # print('test shape:', preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        return

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
    
    def train_rl(self, setting):
        RL_DATA_PATH = f'{self.args.checkpoints}{setting}'
        BUFFER_PATH = f'{RL_DATA_PATH}/buffer/'
        unify_input_data(RL_DATA_PATH)

        (train_X, valid_X, test_X, train_y, valid_y, test_y, train_error, valid_error, _) = load_data(f'{RL_DATA_PATH}/dataset/input_rl.npz')

        train_preds = np.load(f'{RL_DATA_PATH}/rl_bm/bm_train_preds.npy')
        valid_preds = np.load(f'{RL_DATA_PATH}/rl_bm/bm_vali_preds.npy')
        test_preds = np.load(f'{RL_DATA_PATH}/rl_bm/bm_test_preds.npy')

        train_X = np.swapaxes(train_X, 2, 1)
        valid_X = np.swapaxes(valid_X, 2, 1)
        test_X  = np.swapaxes(test_X,  2, 1)

        L = len(train_X) - 1 if self.args.use_td else len(train_X)

        states = torch.FloatTensor(train_X).to(self.device)
        valid_states = torch.FloatTensor(valid_X).to(self.device)
        test_states = torch.FloatTensor(test_X).to(self.device)
        
        obs_dim = train_X.shape[1]          # observation dimension (dataset features)
        act_dim = train_error.shape[-1]     # actor dimension (action dimension)

        env = Env(train_error, train_y, train_preds)
        best_model_weight = get_state_weight(train_error)

        if not os.path.exists(BUFFER_PATH):
            batch_buffer = []
            
            for state_idx in trange(L, desc="[Create buffer]"):
                best_model_idx = train_error[state_idx].argmin()
                
                for action_idx in range(act_dim):
                    rank, mape, mae, mse = env.reward_func(state_idx, action_idx)
                    batch_buffer.append([state_idx, action_idx, rank, mape, mae, mse, best_model_weight[best_model_idx]])
            
            batch_buffer_df = pd.DataFrame(
                batch_buffer,
                columns=['state_idx', 'action_idx', 'rank', 'mape', 'mae', 'mse','best_model_weight']
            )

            os.makedirs(BUFFER_PATH)
            with open(f'{BUFFER_PATH}/batch_buffer.csv', 'w') as f:
                batch_buffer_df.to_csv(f)
        else:
            batch_buffer_df = pd.read_csv(f'{BUFFER_PATH}/batch_buffer.csv', index_col=0)
        
        q_mape = [batch_buffer_df['mape'].quantile(0.1*i) for i in range(1, 10)]
        q_mae = [batch_buffer_df['mae'].quantile(0.1*i) for i in range(1, 10)]
        q_mse = [batch_buffer_df['mse'].quantile(0.1*i) for i in range(1, 10)]

        if self.args.use_td:
            batch_buffer_df = batch_buffer_df.query(f'state_idx < {L}')

        # state weight
        state_weights = [1/best_model_weight[i] for i in train_error.argmin(1)]
        if self.args.use_weight:
            state_weights = torch.FloatTensor(state_weights).to(self.device)
        else:
            state_weights = None

        # initialize the DDPG agent
        agent = DDPGAgent(self.device, self.args.use_td, states, obs_dim, act_dim, self.args.hidden_dim, self.args.learn_rate_RL, self.args.gamma, self.args.tau)
        replay_buffer = ReplayBuffer(self.device, act_dim, max_size=int(1e5))
        extra_buffer = ReplayBuffer(self.device, act_dim, max_size=int(1e5))

        if self.args.use_pretrain:
            pretrained_actor = pretrain_actor(self.device, obs_dim, act_dim, self.args.hidden_dim, states, train_error, best_model_weight, valid_states, valid_error)

            # copy the pretrained actor 
            for param, target_param in zip(
                    pretrained_actor.parameters(), agent.actor.parameters()):
                target_param.data.copy_(param.data)

            for param, target_param in zip(
                    pretrained_actor.parameters(), agent.target_actor.parameters()):
                target_param.data.copy_(param.data)

        # to save the best model
        best_actor = Actor(obs_dim, act_dim, self.args.hidden_dim).to(self.device)
        for param, target_param in zip(agent.actor.parameters(), best_actor.parameters()):
            target_param.data.copy_(param.data)
            
        # warm up
        for _ in trange(self.args.RL_warmup_epoch, desc='[Warm Up]'):
            shuffle_idxes   = np.random.randint(0, L, 300)
            sampled_states  = states[shuffle_idxes] 
            sampled_actions = agent.select_action(sampled_states)
            sampled_rewards, _ = get_batch_reward(env, shuffle_idxes, sampled_actions, q_mape, q_mae, q_mse)

            for i in range(len(sampled_states)):
                replay_buffer.add(shuffle_idxes[i], sampled_actions[i], sampled_rewards[i])
                if self.args.use_extra and sampled_rewards[i] <= -1.:
                    extra_buffer.add(shuffle_idxes[i], sampled_actions[i], sampled_rewards[i])

        step_size = self.args.RL_step_size
        step_num  = int(np.ceil(L / step_size))
        best_mae_loss = np.inf
        patience, max_patience = 0, self.args.RL_max_patience
        epsilon = self.args.epsilon

        for epoch in trange(self.args.RL_epochs):
            t1 = time.time()
            q_loss_lst, pi_loss_lst, q_lst, target_q_lst  = [], [], [], []
            shuffle_idx = np.random.permutation(np.arange(L))

            for i in trange(step_num, desc=f'[Step Num]'):
                batch_idx = shuffle_idx[i*step_size: (i+1)*step_size]        # (512,)
                batch_states = states[batch_idx]
                if np.random.random() < epsilon:
                    batch_actions = sparse_explore(batch_states, act_dim)
                else:
                    batch_actions = agent.select_action(batch_states)
                batch_rewards, _ = get_batch_reward(env, batch_idx, batch_actions, q_mape, q_mae, q_mse)
                
                for j in range(len(batch_idx)):
                    replay_buffer.add(batch_idx[j], batch_actions[j], batch_rewards[j])
                    if self.args.use_extra and batch_rewards[j] <= -1.:
                        extra_buffer.add(batch_idx[j], batch_actions[j], batch_rewards[j])

                sampled_obs_idxes, sampled_actions, sampled_rewards = replay_buffer.sample(512)
                if self.args.use_weight:
                    sampled_weights = state_weights[sampled_obs_idxes]
                else:
                    sampled_weights = None
                info = agent.update(sampled_obs_idxes, sampled_actions, sampled_rewards, sampled_weights)
                pi_loss_lst.append(info['pi_loss'])
                q_loss_lst.append(info['q_loss'])
                q_lst.append(info['current_q'])
                target_q_lst.append(info['target_q'])

                if self.args.use_extra and extra_buffer.ptr > 512:
                    sampled_obs_idxes, sampled_actions, sampled_rewards = extra_buffer.sample(512)
                    if self.args.use_weight:
                        sampled_weights = state_weights[sampled_obs_idxes]
                    else:
                        sampled_weights = None
                    info = agent.update(sampled_obs_idxes, sampled_actions, sampled_rewards, sampled_weights)
                    pi_loss_lst.append(info['pi_loss'])
                    q_loss_lst.append(info['q_loss'])
                    q_lst.append(info['current_q'])
                    target_q_lst.append(info['target_q'])

            valid_mse_loss, valid_mae_loss, valid_mape_loss, count_lst, _, _ = evaluate_agent(agent, valid_states, valid_preds, valid_y)

            print(f'\n# Epoch {epoch + 1} ({(time.time() - t1)/60:.2f} min): '
                f'valid_mse_loss: {valid_mse_loss:.3f}\t'
                f'valid_mae_loss: {valid_mae_loss:.3f}\t'
                f'valid_mape_loss: {valid_mape_loss*100:.3f}\t' 
                f'q_loss: {np.average(q_loss_lst):.5f}\t'
                f'current_q: {np.average(q_lst):.5f}\t'
                f'target_q: {np.average(target_q_lst):.5f}\n')
            
            if not os.path.exists(f"{RL_DATA_PATH}/train_results/rl/"):
                os.makedirs(f'{RL_DATA_PATH}/train_results/rl/')

            log_file = open(f'{RL_DATA_PATH}/train_results/rl/rl_log.txt', 'a')
            log_file.write(f'# Epoch {epoch + 1}:\n'
                f'valid_mse_loss: {valid_mse_loss:.3f}\t'
                f'valid_mae_loss: {valid_mae_loss:.3f}\t'
                f'valid_mape_loss: {valid_mape_loss*100:.3f}\t'
                f'q_loss: {np.average(q_loss_lst):.5f}\t'
                f'current_q: {np.average(q_lst):.5f}\t'
                f'target_q: {np.average(target_q_lst):.5f}\n\n')
            log_file.close()        
            
            if valid_mae_loss < best_mae_loss:
                best_mae_loss = valid_mae_loss
                patience = 0
                # save best model
                for param, target_param in zip(agent.actor.parameters(), best_actor.parameters()):
                    target_param.data.copy_(param.data)
            else:
                patience += 1
            if patience == max_patience:
                break
            epsilon = max(epsilon-0.2, 0.1)

        for param, target_param in zip(agent.actor.parameters(), best_actor.parameters()):
            param.data.copy_(target_param)

        test_mse_loss, test_mae_loss, test_mape_loss, count_lst, pred, true = evaluate_agent(agent, test_states, test_preds, test_y)
        
        # save result
        folder_path = './checkpoints/' + setting + '/testing_results/rl/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        testing_result_path = './checkpoints/' + setting + '/testing_results/'

        print(
            f'test_mse_loss: {test_mse_loss:.3f}\t'
            f'test_mae_loss: {test_mae_loss:.3f}\t'
            f'test_mape_loss: {test_mape_loss*100:.3f}'
            )

        res_file = open(testing_result_path + 'result_RL.txt', 'a')
        res_file.write(
            f'test_mse_loss: {test_mse_loss:.3f}\t'
            f'test_mae_loss: {test_mae_loss:.3f}\t'
            f'test_mape_loss: {test_mape_loss*100:.3f}'
            )
        res_file.write('\n')
        res_file.write('\n')
        res_file.close()

        with open(f'{folder_path}/' + 'count_sorted_act.npy', 'wb') as f:
            np.save(f, count_lst)
        with open(f'{folder_path}/' + 'metrics.npy', 'wb') as f:
            np.save(f, np.array([test_mse_loss, test_mae_loss, test_mape_loss]))
        with open(f'{folder_path}/' + 'pred.npy', 'wb') as f:
            np.save(f, pred)
        with open(f'{folder_path}/' + 'true.npy', 'wb') as f:
            np.save(f, true)
        
        for i in range(len(pred)):
            visual(true[i], pred[i], f'{folder_path}/' + f'{i}.pdf')

        return
