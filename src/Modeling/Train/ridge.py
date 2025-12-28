import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd

from Config.load import load_config
from Enum.features_enums import FeatureEnum as fenum    
from Enum.paths_enums import PathEnum as penum
from Helper.prepare import Preparing
from log.apply_log import log_result
from sklearn.linear_model import Ridge
from Helper.eval import eval_model

config = load_config()
train_val_path = penum.TRAIN_VAL_PATH.value
ridge_parm = config['Model']['Ridge']
random_state = config['RANDOM_STATE']

class Train():
    def __init__(self, x_train, x_val, t_train, t_val):
        self.x_train = x_train
        self.x_val = x_val
        self.t_train =  t_train
        self.t_val = t_val

    def try_ridge(self):
        model = Ridge(alpha=ridge_parm['alpha'], fit_intercept=ridge_parm['fit_intercept'], solver=ridge_parm['solver'], positive=ridge_parm['positive'], random_state=random_state)
        model.fit(self.x_train, self.t_train)

        log_result(f'Train Path', 'Ridge')
        train_score, train_error = eval_model(model, self.x_train, self.t_train, 'train')
        val_score, val_error = eval_model(model, self.x_val, self.t_val, 'val')


        log_result(f"MSE for Train: {train_error}", "Ridge")
        log_result(f"R2Score for Train: {train_score}", "Ridge")
        log_result(f"MSE for Val: {val_error}", "Ridge")
        log_result(f"R2Score for Val: {val_score}", "Ridge")
        log_result(f"Parameters: alpha = {ridge_parm['alpha']}, fit_intercept = {ridge_parm['fit_intercept']}, sover = {ridge_parm['solver']}, positive = {ridge_parm['positive']}", "Ridge")
        log_result('--'*40, "Ridge")

        return model




if __name__ == '__main__':
    prepare = Preparing()
    x_train, x_val, encode_season, encode_store, t_train, t_val, poly, scaler = prepare.prepare_data()
    print(x_train.shape, t_train.shape, x_val.shape, t_val.shape)    

    train = Train(x_train, x_val, t_train, t_val)
    model = train.try_ridge()
    # eval_model(model, x_train, t_train, 'train')
    # eval_model(model, x_val, t_val, 'val')


    print("Successful")


