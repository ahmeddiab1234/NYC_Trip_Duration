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
from sklearn.neural_network import MLPRegressor
from Helper.eval import eval_model

config = load_config()
train_val_path = penum.TRAIN_VAL_PATH.value
neural_network_parm = config['Model']['NeuralNetwork']
random_state = config['RANDOM_STATE']

class Train():
    def __init__(self, x_train, x_val, t_train, t_val):
        self.x_train = x_train
        self.x_val = x_val
        self.t_train =  t_train
        self.t_val = t_val

    def try_neural_network(self):
        model = MLPRegressor(
            hidden_layer_sizes=neural_network_parm['hidden_layer'],
            solver= neural_network_parm['solver'],
            learning_rate= 'adaptive',
            learning_rate_init= neural_network_parm['init_lr'],
            alpha= neural_network_parm['alpha'],
            max_iter= neural_network_parm['max_iter'],
            early_stopping= neural_network_parm['early_stopping'],
            random_state = random_state
        )

        log_result(f'Train Path', 'NeuralNetwork')
        model.fit(self.x_train, self.t_train)
        train_score, train_error = eval_model(model, self.x_train, self.t_train, 'train')
        val_score, val_error = eval_model(model, self.x_val, self.t_val, 'val')


        log_result(f"MSE for Train: {train_error}", "NeuralNetwork")
        log_result(f"R2Score for Train: {train_score}", "NeuralNetwork")
        log_result(f"MSE for Val: {val_error}", "NeuralNetwork")
        log_result(f"R2Score for Val: {val_score}", "NeuralNetwork")

        param = f" \
            hidden_layers , {neural_network_parm['hidden_layer']}  \
            solver , {neural_network_parm['solver']}  \
            init_lr , {neural_network_parm['init_lr']}  \
            max_iter , {neural_network_parm['max_iter'] } \
            early_stopping , {neural_network_parm['early_stopping']}  \
            alpha , {neural_network_parm['alpha']}  \
        "
        log_result(str(param), 'NeuralNetwork')
        log_result('--'*40, "NeuralNetwork")

        return model




if __name__ == '__main__':
    prepare = Preparing()
    x_train, x_val, encode_season, encode_store, t_train, t_val, poly, scaler = prepare.prepare_data()
    print(x_train.shape, t_train.shape, x_val.shape, t_val.shape)    

    train = Train(x_train, x_val, t_train, t_val)
    model = train.try_neural_network()
    # eval_model(model, x_train, t_train, 'train')
    # eval_model(model, x_val, t_val, 'val')


    print("Successful")


