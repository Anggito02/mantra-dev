import gym
from gym import spaces
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class OpenNetEnv(gym.Env):
    def __init__(self, data_path, n_learner, mode='train'):
        super(OpenNetEnv, self).__init__()
        self.data_path = data_path
        self.n_learner = n_learner
        self.mode = mode
        self.current_step = 0

        # data
        self.data = self.__getdata__()
        self.X = self.data[0]
        self.y = self.data[1]
        self.preds = self.data[2]
        self.n_steps = len(self.X)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_learner,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.X.shape[0], dtype=np.float32)

    def __getdata__(self):
        mode_X = np.load(f'{self.data_path}/dataset/input_{self.mode}_x.npy')
        mode_y = np.load(f'{self.data_path}/dataset/input_{self.mode}_y.npy')
        
        mode_bm_preds_npz = np.load(f'{self.data_path}/rl_bm/bm_{self.mode}_preds.npz')
        merged_preds = []
        for model_name in mode_bm_preds_npz.keys():
            preds = mode_bm_preds_npz[model_name]
            preds = np.expand_dims(preds, axis=1)
            merged_preds.append(preds)
        merged_preds = np.concatenate(merged_preds, axis=1)

        mode_X = mode_X.reshape(-1)
        mode_y = mode_y.reshape(-1)
        merged_preds = merged_preds.reshape(-1, merged_preds.shape[1])
        
        return [mode_X, mode_y, merged_preds]

    def reset(self):
        self.current_step = 0
        return self.X[self.current_step]

    def step(self, action):
        # Ensure the action is within the expected range
        action = np.clip(action, 0, 1)
        
        # Normalize the weights to sum to 1
        weights = action / np.sum(action)

        # Combine predictions using the weights
        selected_pred = np.sum(self.preds[self.current_step] * weights[:, 1], axis=0)

        # Calculate reward using MSE and MAE
        mse = mean_squared_error(self.y[self.current_step].flatten(), selected_pred.flatten())
        mae = mean_absolute_error(self.y[self.current_step].flatten(), selected_pred.flatten())
        reward = - (mse + mae)  # Negative because we want to minimize the error

        self.current_step += 1
        done = self.current_step >= self.n_steps
        
        if done:
            next_state = self.X[self.current_step - 1]
        else:
            next_state = self.X[self.current_step]

        info = {
            'mse': mse,
            'mae': mae
        }

        return next_state, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

def evaluate_policy(model, env, n_eval_episodes=10):
    """
    Evaluates the policy by running it for a certain number of episodes.

    :param model: The RL model to be evaluated.
    :param env: The environment to evaluate the model on.
    :param n_eval_episodes: The number of episodes to evaluate the model.
    :return: The mean reward and standard deviation of the reward over the evaluated episodes.
    """
    episode_rewards = []
    total_mse_loss = []
    total_mae_loss = []

    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        mse_loss = 0
        mae_loss = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            mse_loss += info['mse']
            mae_loss += info['mae']
            total_reward += reward

        episode_rewards.append(total_reward)
        total_mse_loss.append(mse_loss)
        total_mae_loss.append(mae_loss)

    total_mse_loss = np.mean(total_mse_loss)
    total_mae_loss = np.mean(total_mae_loss)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return total_mse_loss, total_mae_loss, mean_reward, std_reward