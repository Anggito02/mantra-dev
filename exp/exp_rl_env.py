import numpy as np
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

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
