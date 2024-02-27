import os
import numpy as np
import pandas as pd
import torch
from tqdm import trange
import time

from exp.exp_basic import Exp_Basic
from exp.exp_rl_pretrain import Exp_RL_Pretrain
from exp.exp_rl_env import Env
from utils.utils import unify_input_data, load_data, get_state_weight, get_batch_reward, sparse_explore, evaluate_agent

from models.ddpg import Actor, DDPGAgent, ReplayBuffer

class OPT_RL_Mantra(Exp_Basic):
    def __init__(self, args, setting):
        self.args = args
        self.setting = setting
        self.device = self._acquire_device()
        self.RL_DATA_PATH = f'{args.checkpoints}{setting}'
        self.BUFFER_PATH = f'{self.RL_DATA_PATH}/buffer/'

    def forward(self):
        unify_input_data(self.RL_DATA_PATH)

        (train_X, valid_X, test_X, train_y, valid_y, test_y, train_error, valid_error, _) = load_data(f'{self.RL_DATA_PATH}/dataset/input_rl.npz')

        valid_preds = np.load(f'{self.RL_DATA_PATH}/rl_bm/bm_vali_preds.npy')
        test_preds = np.load(f'{self.RL_DATA_PATH}/rl_bm/bm_test_preds.npy')

        train_X = np.swapaxes(train_X, 2, 1)
        valid_X = np.swapaxes(valid_X, 2, 1)
        test_X  = np.swapaxes(test_X,  2, 1)

        L = len(train_X) - 1 if self.args.use_td else len(train_X)

        states = torch.FloatTensor(train_X).to(self.device)
        valid_states = torch.FloatTensor(valid_X).to(self.device)
        test_states = torch.FloatTensor(test_X).to(self.device)
        
        obs_dim = train_X.shape[1]          # observation dimension (dataset features)
        act_dim = train_error.shape[-1]     # actor dimension (action dimension)

        env = Env(train_error, train_y, self.RL_DATA_PATH)
        best_model_weight = get_state_weight(train_error)

        if not os.path.exists(self.BUFFER_PATH):
            batch_buffer = []
            
            for state_idx in trange(L, desc="[Create buffer]"):
                best_model_idx = train_error[state_idx].argmin()
                
                for action_idx in range(act_dim):
                    rank, mape, mae = env.reward_func(state_idx, action_idx)
                    batch_buffer.append([state_idx, action_idx, rank, mape, mae, best_model_weight[best_model_idx]])
            
            batch_buffer_df = pd.DataFrame(
                batch_buffer,
                columns=['state_idx', 'action_idx', 'rank', 'mape', 'mae', 'best_model_weight']
            )

            os.makedirs(self.BUFFER_PATH)
            batch_buffer_df.to_csv(f'{self.BUFFER_PATH}/batch_buffer.csv')
        else:
            batch_buffer_df = pd.read_csv(f'{self.BUFFER_PATH}/batch_buffer.csv', index_col=0)
        
        q_mape = [batch_buffer_df['mape'].quantile(0.1*i) for i in range(1, 10)]
        # q_mae = [batch_buffer_df['mape'].quantile(0.1*i) for i in range(1, 10)]

        if self.args.use_td:
            batch_buffer_df = batch_buffer_df.query(f'state_idx < {L}')

        # state weight
        state_weights = [1/best_model_weight[i] for i in train_error.argmin(1)]
        if self.args.use_weight:
            state_weights = torch.FloatTensor(state_weights).to(self.device)
        else:
            state_weights = None

        # initialize the DDPG agent
        agent = DDPGAgent(self.args, states, obs_dim, act_dim, self.args.hidden_dim, self.device)
        replay_buffer = ReplayBuffer(self.args, self.device, act_dim, max_size=int(1e5))
        extra_buffer = ReplayBuffer(self.args, self.device, act_dim, max_size=int(1e5))

        if self.args.use_pretrain:
            exp_pretrain = Exp_RL_Pretrain(
                self.args, self.device, obs_dim, act_dim, self.args.hidden_dim, states=states, train_error=train_error, cls_weights=state_weights, valid_states=valid_states, valid_error=valid_error
            )

            pretrained_actor = exp_pretrain.forward()

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
        for _ in trange(self.args.RL_warmup_epochs, desc='[Warm Up]'):
            shuffle_idxes   = np.random.randint(0, L, 300)
            sampled_states  = states[shuffle_idxes] 
            sampled_actions = agent.select_action(sampled_states)
            sampled_rewards, _ = get_batch_reward(env, shuffle_idxes, sampled_actions, q_mape)

            for i in range(len(sampled_states)):
                replay_buffer.add(shuffle_idxes[i], sampled_actions[i], sampled_rewards[i])
                if self.args.use_extra and sampled_rewards[i] <= -1.:
                    extra_buffer.add(shuffle_idxes[i], sampled_actions[i], sampled_rewards[i])

        step_size = self.args.RL_step_size
        step_num  = int(np.ceil(L / step_size))
        best_mape_loss = np.inf
        patience, max_patience = 0, self.args.RL_max_patience
        epsilon = self.args.epsilon

        for epoch in trange(self.args.RL_epochs):
            t1 = time.time()
            q_loss_lst, pi_loss_lst, q_lst, target_q_lst  = [], [], [], []
            shuffle_idx = np.random.permutation(np.arange(L))

            for i in trange(step_num, desc=f'[Step Num]'):
                batch_idx = shuffle_idx[i*step_size: (i+1)*step_size]        # (512,)
                batch_states = states[batch_idx]
                if np.random.random() < self.args.epsilon:
                    batch_actions = sparse_explore(batch_states, act_dim)
                else:
                    batch_actions = agent.select_action(batch_states)
                batch_rewards, batch_mae = get_batch_reward(env, batch_idx, batch_actions, q_mape)
                
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

            valid_mae_loss, valid_mape_loss, count_lst = evaluate_agent(agent, valid_states, valid_preds, valid_y)
            print(f'\n# Epoch {epoch + 1} ({(time.time() - t1)/60:.2f} min): '
                f'valid_mae_loss: {valid_mae_loss:.3f}\t'
                f'valid_mape_loss: {valid_mape_loss*100:.3f}\t' 
                f'q_loss: {np.average(q_loss_lst):.5f}\t'
                f'current_q: {np.average(q_lst):.5f}\t'
                f'target_q: {np.average(target_q_lst):.5f}\n')
            
            if valid_mape_loss < best_mape_loss:
                best_mape_loss = valid_mape_loss
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

        test_mae_loss, test_mape_loss, count_lst = evaluate_agent(
            agent, test_states, test_preds, test_y)
        
        print(f'test_mae_loss: {test_mae_loss:.3f}\t'
            f'test_mape_loss: {test_mape_loss*100:.3f}')
        
        if not os.path.exists(self.RL_DATA_PATH):
            os.makedirs(f'{self.RL_DATA_PATH}/result/')

        res_file = open(f'{self.RL_DATA_PATH}/result_RL.txt', 'a')
        res_file.write(f'test_mae_loss: {test_mae_loss:.3f}\t'
            f'test_mape_loss: {test_mape_loss*100:.3f}')
        res_file.write('\n')
        res_file.write('\n')
        res_file.close()
        
        return