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
from xgboost import XGBRegressor
from Helper.eval import eval_model
from Helper.save import save_model
from Enum.model_enums import ModelEnum as menum

config = load_config()
train_val_path = penum.TRAIN_VAL_PATH.value
xgboost_parm = config['Model']['XGBoost']
random_state = config['RANDOM_STATE']
model_name = menum.XGBOOST.value

class Train():
    def __init__(self, x_train, x_val, t_train, t_val):
        self.x_train = x_train
        self.x_val = x_val
        self.t_train =  t_train
        self.t_val = t_val

    def try_xgboost(self):
        model = XGBRegressor(
            n_estimators = xgboost_parm['n_estimators'],
            learning_rate = xgboost_parm['learning_rate'],
            max_depth = xgboost_parm['max_depth'],
            min_child_weight = xgboost_parm['min_child_weigth'],
            gamma = xgboost_parm['gamma'],
            reg_alpha = xgboost_parm['reg_alpha'],
            reg_lambda = xgboost_parm['reg_lambda'],
            random_state = random_state
        )

        log_result(f'Train Path', 'Xgboost')
        model.fit(self.x_train, self.t_train)
        train_score, train_error = eval_model(model, self.x_train, self.t_train, 'train')
        val_score, val_error = eval_model(model, self.x_val, self.t_val, 'val')


        log_result(f"MSE for Train: {train_error}", "Xgboost")
        log_result(f"R2Score for Train: {train_score}", "Xgboost")
        log_result(f"MSE for Val: {val_error}", "Xgboost")
        log_result(f"R2Score for Val: {val_score}", "Xgboost")

        param = " \
            n_estimators , {xgboost_parm['n_estimators']} \n \
            learning_rate , {xgboost_parm['learning_rate']} \n \
            max_depth , {xgboost_parm['max_depth']} \n \
            min_child_weight , {xgboost_parm['min_child_weigth']} \n \
            gamma , {xgboost_parm['gamma']} \n \
            reg_alpha , {xgboost_parm['reg_alpha']} \n \
            reg_lambda , {xgboost_parm['reg_lambda']} \n \
        "
        log_result(param, 'Xgboost')
        log_result('--'*40, "Xgboost")

        return model



if __name__ == '__main__':
    prepare = Preparing()
    x_train, x_val, encode_season, encode_store, t_train, t_val, poly, scaler = prepare.prepare_data()
    print(x_train.shape, t_train.shape, x_val.shape, t_val.shape)    

    train = Train(x_train, x_val, t_train, t_val)
    model = train.try_xgboost()
    print("Successful")


