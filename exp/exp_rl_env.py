import numpy as np
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_absolute_percentage_error

class Env:
    def __init__(self, train_error, train_y):
        self.error = train_error
        # self.bm_preds = np.load(f'{DATA_DIR}/bm_train_preds.npy')
        self.y = train_y
        
    def reward_func(self, idx, action):
        def inv_trans(x): 
            # SCALE_MEAN, SCALE_STD = np.load(f'dataset/scaler.npy')
            SCALE_STD = 1.0
            SCALE_MEAN = 0.0
            return x * SCALE_STD + SCALE_MEAN
    
        if isinstance(action, int):
            tmp = np.zeros(self.bm_preds.shape[1])
            tmp[action] = 1.
            action = tmp
        weighted_y = np.multiply(action.reshape(-1, 1), self.bm_preds[idx])
        weighted_y = weighted_y.sum(axis=0)
        new_mape = mean_absolute_percentage_error(inv_trans(self.y[idx]), inv_trans(weighted_y))
        new_mae = mean_absolute_error(inv_trans(self.y[idx]), inv_trans(weighted_y))
        new_error = np.array([*self.error[idx], new_mape])
        rank = np.where(np.argsort(new_error) == len(new_error) - 1)[0][0]
        return rank, new_mape, new_mae 
