import torch

import numpy as np
import pandas as pd
from tqdm import trange
from collections import Counter

from sktime.performance_metrics.forecasting import \
    mean_absolute_error, mean_absolute_percentage_error

DATA_DIR = 'dataset'
SCALE_MEAN, SCALE_STD = np.load(f'{DATA_DIR}/scaler.npy')
def inv_trans(x): return x * SCALE_STD + SCALE_MEAN

def get_state_weight(train_error):
    L = len(train_error)
    best_model = train_error.argmin(1)
    best_model_counter = Counter(best_model)
    model_weight = {k:v/L for k,v in best_model_counter.items()}
    return model_weight

def get_model_info(error_array):
    model_rank = np.zeros_like(error_array)
    sort_res   = error_array.argsort(1)
    model_rank[1:] = sort_res[:-1]
    model_rank[0]  = sort_res[-1]
    return model_rank


def compute_mape_error(y, bm_preds):
    mape_loss_df = pd.DataFrame()
    for i in trange(bm_preds.shape[1], desc='[Compute Error]'):
        model_mape_loss = [mean_absolute_percentage_error(
            inv_trans(y[j]), inv_trans(bm_preds[j, i, :]),
            symmetric=True) for j in range(len(y))]
        mape_loss_df[i] = model_mape_loss
    return mape_loss_df


def compute_mae_error(y, bm_preds):
    loss_df = pd.DataFrame()
    for i in trange(bm_preds.shape[1], desc='[Compute Error]'):
        model_mae_loss = [mean_absolute_error(
            inv_trans(y[j]), inv_trans(bm_preds[j, i, :]),
            symmetric=True) for j in range(len(y))]
        loss_df[i] = model_mae_loss
    return loss_df

def sparse_explore(obs, act_dim):
    N = len(obs)
    x = np.zeros((N, act_dim))
    randn_idx = np.random.randint(0, act_dim, size=(N,))
    x[np.arange(N), randn_idx] = 1

    # disturb from the vertex
    delta = np.random.uniform(0.02, 0.1, size=(N, 1))
    x[np.arange(N), randn_idx] -= delta.squeeze()

    # noise
    noise = np.abs(np.random.randn(N, act_dim))
    noise[np.arange(N), randn_idx] = 0
    noise /= noise.sum(1, keepdims=True)
    noise = delta * noise
    sparse_action = x + noise

    return sparse_action

def unify_input_data():
    train_val_X   = np.load('dataset/train_val_X.npy')    # (62795, 120, 7)
    train_val_y   = np.load('dataset/train_val_y.npy')    # (62795, 24)
    test_X        = np.load('dataset/test_X.npy')         # (6867, 120, 7)
    test_y        = np.load('dataset/test_y.npy')         # (6867, 24)

    L = len(test_y)
    train_X = train_val_X[:-L]
    valid_X = train_val_X[-L:]
    train_y = train_val_y[:-L]
    valid_y = train_val_y[-L:]

    # predictions
    MODEL_NAMES = ['lstm1', 'lstm2', 'gru1', 'gru2', 'cnn1', 'cnn2',
                   'transformer1', 'transformer2', 'repeat']
    merge_data = []
    bm_train_preds = np.load('dataset/bm_train_preds.npz')
    for model_name in MODEL_NAMES:
        model_pred = bm_train_preds[model_name]
        model_pred = np.expand_dims(model_pred, axis=1)
        merge_data.append(model_pred)
    merge_data = np.concatenate(merge_data, axis=1)  # (62795, 9, 24)
    train_preds = merge_data[:-L]
    valid_preds = merge_data[-L:]
    np.save('dataset/bm_train_preds.npy', train_preds)
    np.save('dataset/bm_valid_preds.npy', valid_preds)

    merge_test_data = []
    bm_train_preds = np.load('dataset/bm_test_preds.npz')
    for model_name in MODEL_NAMES:
        model_pred = bm_train_preds[model_name]
        model_pred = np.expand_dims(model_pred, axis=1)
        merge_test_data.append(model_pred)
    test_preds = np.concatenate(merge_test_data, axis=1)  # (62795, 9, 24)
    np.save('dataset/bm_test_preds.npy', test_preds)

    train_error_df = compute_mape_error(train_y, train_preds)
    valid_error_df = compute_mape_error(valid_y, valid_preds)
    test_error_df  = compute_mape_error(test_y , test_preds)

    np.savez('dataset/input.npz',
             train_X=train_X,
             valid_X=valid_X,
             test_X=test_X,
             train_y=train_y,
             valid_y=valid_y,
             test_y=test_y,
             train_error=train_error_df,
             valid_error=valid_error_df,
             test_error=test_error_df
            )


def load_data():
    input_data = np.load('dataset/input.npz')
    train_X = input_data['train_X']
    valid_X = input_data['valid_X']
    test_X  = input_data['test_X' ]
    train_y = input_data['train_y']
    valid_y = input_data['valid_y']
    test_y  = input_data['test_y' ]
    print("First Get: ", train_X.shape, valid_X.shape, test_X.shape)
    
    train_error = input_data['train_error']  # (55928, 9)
    valid_error = input_data['valid_error']  # (6867,  9)
    test_error  = input_data['test_error' ]  # (6867,  9)
    print("Errors: ", train_error.shape, valid_error.shape, test_error.shape)
    # exit()
    return (train_X, valid_X, test_X, train_y, valid_y, test_y,
            train_error, valid_error, test_error)


def plot_best_data(train_error, valid_error, test_error):
    import matplotlib.pyplot as plt
    train_min_model = Counter(train_error.argmin(1))
    valid_min_model = Counter(valid_error.argmin(1))
    test_min_model  = Counter(test_error.argmin(1))

    train_best_num = [train_min_model[i] for i in range(9)]
    valid_best_num = [valid_min_model[i] for i in range(9)]
    test_best_num  = [test_min_model[i] for i in range(9)]

    labels  = [f'M{i}' for i in range(1, 10)]

    fig, ax = plt.subplots()
    width = 0.3
    x = np.arange(len(labels)) * 1.5
    _ = ax.bar(x - width, train_best_num, width, label='train')
    _ = ax.bar(x, valid_best_num, width, label='valid')
    _ = ax.bar(x + width, test_best_num, width, label='test')

    ax.set_title('Jena Climate Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.savefig('jena_best_model.png', dpi=300)

# mape reward computed by the quantile
def get_mape_reward(q_mape, mape):
    q = 0
    while (q < 9) and (mape > q_mape[q]):
        q += 1
    reward = 1 - 2 * q / 9
    return reward


# mae reward computed by the quantile
def get_mae_reward(q_mae, mae):
    q = 0
    while (q < 9) and (mae > q_mae[q]):
        q += 1
    reward = 1 - 2 * q / 9
    return reward

# rank reward
def get_rank_reward(rank):
    reward = 1 - 2 * rank / 9
    return reward

def get_batch_reward(env, idxes, actions, q_mape, q_mae=None):
    rewards = []
    mae_lst = []
    for i in range(len(idxes)):
        rank, new_mape, new_mae = env.reward_func(idxes[i], actions[i])
        rank_reward = get_rank_reward(rank, 1)
        mape_reward = get_mape_reward(q_mape, new_mape, 1)
        # mae_reward  = get_mae_reward(q_mae, new_mae, 2)
        
        combined_reward = mape_reward + rank_reward # + mae_reward
        mae_lst.append(new_mae)
        rewards.append(combined_reward)
    return rewards, mae_lst

############
# evaluate #
############
def evaluate_agent(agent, test_states, test_bm_preds, test_y):
    with torch.no_grad():
        weights = agent.select_action(test_states)  # (2816, 9)
    act_counter = Counter(weights.argmax(1))
    act_sorted  = sorted([(k, v) for k, v in act_counter.items()])
    weights = np.expand_dims(weights, -1)  # (2816, 9, 1)
    weighted_y = weights * test_bm_preds  # (2816, 9, 24)
    weighted_y = weighted_y.sum(1)  # (2816, 24)
    mae_loss = mean_absolute_error(inv_trans(test_y), inv_trans(weighted_y))
    mape_loss = mean_absolute_percentage_error(inv_trans(test_y), inv_trans(weighted_y))
    return mae_loss, mape_loss, act_sorted
