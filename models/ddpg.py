import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import softmax

from torch.optim import Adam
from models.causal_cnn import CausalCNNEncoder
from exp.exp_basic import Exp_Basic

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=100):
        super().__init__()
        self.cnn_encoder = CausalCNNEncoder(depth=3,
                                            kernel_size=3,
                                            in_channels=obs_dim,
                                            channels=40,
                                            out_channels=hidden_dim,
                                            reduced_size=hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
    
    def forward(self, obs):
        x = F.relu(self.cnn_encoder(obs))
        x = self.net(x)
        return x


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=100):
        super().__init__()
        self.cnn_encoder = CausalCNNEncoder(depth=3,
                                            kernel_size=3,
                                            in_channels=obs_dim,
                                            channels=40,
                                            out_channels=hidden_dim,
                                            reduced_size=hidden_dim)
        self.act_layer = nn.Linear(act_dim, hidden_dim)
        self.fc_layer = nn.Linear(hidden_dim, 1)
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, obs, act):
        x = F.relu(self.cnn_encoder(obs) + self.act_layer(act))
        x = self.net(x)
        return x.squeeze()

class ReplayBuffer():
    def __init__(self, args, device, action_dim, max_size=int(1e5)):
        self.args = args
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

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

class DDPGAgent():
    def __init__(self, args, states, obs_dim, act_dim, hidden_dim, device) -> None:
        super().__init__()
        self.args = args
        self.device = device

        # Initialize actor and target actor
        self.actor = Actor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.target_actor = Actor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.args.learn_rate_RL)

        # Initialize critic
        self.critic = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.target_critic = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.args.learn_rate_RL)

        # Training states
        self.states = states

        # Parameters
        self.gamma = self.args.gamma
        self.tau = self.args.tau
        self.use_td = self.args.use_td

        # Update the target network
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
    
    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])