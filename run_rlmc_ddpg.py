import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

from utils.open_net_env import OpenNetEnv, evaluate_policy

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=False, default='ILI_36_24', help='model id')
    parser.add_argument('--model', type=str, required=False, default='B6iFast',
                        help='model name, options: [Autoformer, Informer, Transformer]')
    parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--seq_len', type=int, default=36, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=18, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--des', type=str, default='normal_0_new', help='description of model')

    parser.add_argument('--n_learner', type=int, default=1, help='num of learners')
    args = parser.parse_args()
    print(f'Exp args:\n{vars(args)}\n')

    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.e_layers,
                args.d_layers,
                args.des)
    
    DATA_DIR = f'./checkpoints/{setting}'    

    # RL training
    print("Training RL Agent...\n")
    train_env = OpenNetEnv(DATA_DIR, args.n_learner, mode="train")
    n_actions = args.n_learner
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", 
                train_env,
                action_noise=action_noise,
                verbose=1,
                buffer_size=10000)
    model.learn(total_timesteps=train_env.n_steps, progress_bar=True)

    # RL validation
    print("Validating RL Agent...\n")
    vali_env = OpenNetEnv(DATA_DIR, args.n_learner, mode="vali")

    mse_loss, mae_loss, mean_reward, std_reward = evaluate_policy(model, vali_env, n_eval_episodes=1)
    print(f"\nRL Validation\nmean_reward={mean_reward:.2f} +/- {std_reward:.2f}\nMSE={mse_loss:.3f}, MAE={mae_loss:.3f}")

    model.save(f"{DATA_DIR}/rl_model")
    del model, train_env, vali_env
    
    # RL testing
    print("Testing RL Agent...\n")
    model = DDPG.load(f"{DATA_DIR}/rl_model")
    test_env = OpenNetEnv(DATA_DIR, args.n_learner, mode="test")

    mse_loss, mae_loss, mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=1)
    print(f"\nRL Test\nmean_reward={mean_reward:.2f} +/- {std_reward:.2f}\nMSE={mse_loss:.3f}, MAE={mae_loss:.3f}")