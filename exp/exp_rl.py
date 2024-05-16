import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm import trange
import time
from collections import Counter
from scipy.special import softmax
from sktime.performance_metrics.forecasting import \
    mean_absolute_error, mean_absolute_percentage_error

from exp.exp_basic import Exp_Basic
from utils.utils import unify_input_data, load_data, get_state_weight, get_batch_reward, sparse_explore, evaluate_agent
from utils.tools import visual

from models.ddpg import Actor, Critic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDPGAgent:
    def __init__(self, use_td, states, obs_dim, act_dim, hidden_dim=256,
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
        new_error = np.array([*self.error[idx], new_mape])
        rank = np.where(np.argsort(new_error) == len(new_error) - 1)[0][0]
        return rank, new_mape, new_mae 


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
    
def pretrain_actor(obs_dim, act_dim, hidden_dim, states, train_error, cls_weights, 
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

class OPT_RL_Mantra:
    def __init__(self, args, setting):
        self.args = args
        self.setting = setting
        self.device = device
        self.RL_DATA_PATH = f'{args.checkpoints}{setting}'
        self.BUFFER_PATH = f'{self.RL_DATA_PATH}/buffer/'

    def forward(self, setting):
        unify_input_data(self.RL_DATA_PATH)

        (train_X, valid_X, test_X, train_y, valid_y, test_y, train_error, valid_error, _) = load_data(f'{self.RL_DATA_PATH}/dataset/input_rl.npz')

        train_preds = np.load(f'{self.RL_DATA_PATH}/rl_bm/bm_train_preds.npy')
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

        env = Env(train_error, train_y, train_preds)
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
            with open(f'{self.BUFFER_PATH}/batch_buffer.csv', 'w') as f:
                batch_buffer_df.to_csv(f)
        else:
            batch_buffer_df = pd.read_csv(f'{self.BUFFER_PATH}/batch_buffer.csv', index_col=0)
        
        q_mape = [batch_buffer_df['mape'].quantile(0.1*i) for i in range(1, 10)]
        q_mae = [batch_buffer_df['mae'].quantile(0.1*i) for i in range(1, 10)]

        if self.args.use_td:
            batch_buffer_df = batch_buffer_df.query(f'state_idx < {L}')

        # state weight
        state_weights = [1/best_model_weight[i] for i in train_error.argmin(1)]
        if self.args.use_weight:
            state_weights = torch.FloatTensor(state_weights).to(self.device)
        else:
            state_weights = None

        # initialize the DDPG agent
        agent = DDPGAgent(self.args.use_td, states, obs_dim, act_dim, self.args.hidden_dim, self.args.learn_rate_RL, self.args.gamma, self.args.tau)
        replay_buffer = ReplayBuffer(self.device, act_dim, max_size=int(1e5))
        extra_buffer = ReplayBuffer(self.device, act_dim, max_size=int(1e5))

        if self.args.use_pretrain:
            pretrained_actor = pretrain_actor(obs_dim, act_dim, self.args.hidden_dim, states, train_error, best_model_weight, valid_states, valid_error)

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
            sampled_rewards, _ = get_batch_reward(env, shuffle_idxes, sampled_actions, q_mape, q_mae)

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
                if np.random.random() < self.args.epsilon:
                    batch_actions = sparse_explore(batch_states, act_dim)
                else:
                    batch_actions = agent.select_action(batch_states)
                batch_rewards, _ = get_batch_reward(env, batch_idx, batch_actions, q_mape, q_mae)
                
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
            
            if not os.path.exists(f"{self.RL_DATA_PATH}/train_results/rl/"):
                os.makedirs(f'{self.RL_DATA_PATH}/train_results/rl/')

            log_file = open(f'{self.RL_DATA_PATH}/train_results/rl/rl_log.txt', 'a')
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

        test_mse_loss, test_mae_loss, test_mape_loss, count_lst, pred, true = evaluate_agent(
            agent, test_states, test_preds, test_y)
        
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