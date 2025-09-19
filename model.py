import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from utils.helper_fun import *
from Processing.preprocessing import Preprocessing_Pipeling
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor as nnr
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

config = load_config()

RANDOM_STATE = config['RANDOME_STATE']
TRAIN_PATH = config['dataset']['train']
VAL_PATH = config['dataset']['val']
TRAIN_VAL = config['dataset']['train_val']

MODEL_NAME = config['Model']
linear_regression_parm = config['Model']['LinearRegression']
ridge_parm = config['Model']['Ridge']
neural_network_parm = config['Model']['NeuralNetwork']
xgboost_parm = config['Model']['XGBoost']


class PrepareData():
    def __init__(self, path=TRAIN_PATH):
        self.df = load_df(path)
        self.train,self.val = split_train_val(self.df)
        self.preprocess_pipeline = Preprocessing_Pipeling()

    def prepare_data(self):
        self.train, encode_season, encode_store = self.preprocess_pipeline.fit_transform(self.train)
        self.val = self.preprocess_pipeline.transform(self.val, encode_season, encode_store)

        target = 'log_trip_duration' if 'log_trip_duration' in self.train.columns else 'trip_duration'
        t_train = self.train[target]
        x_train = self.train.drop(columns=['log_trip_duration','trip_duration'], errors='ignore')
        t_val = self.val[target]
        x_val = self.val.drop(columns=['log_trip_duration','trip_duration'], errors='ignore')

        poly, x_train, x_val = self.preprocess_pipeline.polynomial_feature(x_train, x_val)

        scaler, x_train, x_val = self.preprocess_pipeline.scaling(x_train, x_val)

        return x_train, x_val, encode_season, encode_store, t_train, t_val, poly, scaler
    

class Train():
    def __init__(self,x_train, x_val, t_train, t_val):
        self.x_train=x_train
        self.x_val=x_val
        self.t_train=t_train
        self.t_val=t_val

    def try_linear_regression(self):
        model = LinearRegression(fit_intercept=linear_regression_parm['fit_intercept'])
        model.fit(self.x_train, self.t_train)
        log_result(f'Training Path', 'LinearRegression')
        # log_result(f'Weights: {model.coef_}')
        # log_result(f'Intercept: {model.intercept_}')

        train_score,train_error = eval_model(model,self.x_train,self.t_train, 'train')
        val_score,val_error = eval_model(model,self.x_val,self.t_val, 'val')
        log_result(f'MSE for Train: {train_error}','LinearRegression')
        log_result(f'R2-score for Train: {train_score}','LinearRegression')
        log_result(f'MSE for Val: {val_error}','LinearRegression')
        log_result(f'R2-score for Val: {val_score}','LinearRegression')
        log_result('--'*40,'LinearRegression')
        return model 


    def try_ridge(self):

        log_result(f'Training Path', 'Ridge')
        model = Ridge(alpha=ridge_parm['alpha'], fit_intercept=ridge_parm['fit_intercept'], solver=ridge_parm['solver'], positive=ridge_parm['positive'], random_state=RANDOM_STATE)
        model.fit(self.x_train,self.t_train)

        train_score,train_mse =eval_model(model,self.x_train,self.t_train, 'train')
        val_score, val_mse = eval_model(model,self.x_val,self.t_val, 'val')

        log_result(f'MSE for Train: {train_mse}', name='Ridge')
        log_result(f'R2-score for Train: {train_score}', name='Ridge')
        log_result(f'MSE for Val: {val_mse}', name='Ridge')
        log_result(f'R2-score for Val: {val_score}', name='Ridge')
        log_result(f'parameters: alpha = {ridge_parm['alpha']}, fit-intercept = {ridge_parm['fit_intercept']}', name='Ridge')
        log_result('--'*40, name='Ridge')

        return model

    def try_neural_network(self):
        log_result(f'Training Path', 'Neural_Network')
        model = nnr(
            hidden_layer_sizes=neural_network_parm['hidden_layers'],
            activation='identity',
            solver=neural_network_parm['solver'],
            learning_rate='adaptive',
            learning_rate_init=neural_network_parm['init_lr'],
            alpha=neural_network_parm['alpha'],
            max_iter=neural_network_parm['max_iter'],
            early_stopping=neural_network_parm['early_stopping'],
            random_state=RANDOM_STATE
        )

        model.fit(self.x_train, self.t_train)
        train_score, train_error = eval_model(model, self.x_train, self.t_train, 'train')
        val_score, val_error = eval_model(model, self.x_val, self.t_val, 'val')

        log_result(f'Score train: {train_score}', name='Neural_Network')
        log_result(f'MSE train: {train_error}', name='Neural_Network')
        log_result(f'Score val: {val_score}', name='Neural_Network')
        log_result(f'MSE val: {val_error}', name='Neural_Network')

        current_params = {
            'hidden_layers': neural_network_parm['hidden_layers'],
            'solver': neural_network_parm['solver'],
            'learning_rate_init': neural_network_parm['init_lr'],
            'max_iter': neural_network_parm['max_iter'],
            'early_stopping': neural_network_parm['early_stopping'],
            'alpha': neural_network_parm['alpha']
        }

        log_result(f'Parameters: {current_params}', name='Neural_Network')
        log_result('--'*40, name='Neural_Network')

        return model


    def try_xgboost(self):
        log_result(f'Training Validaion path', 'XGboost')

        model = XGBRegressor(
            n_estimators=xgboost_parm['n_estimators'],
            learning_rate=xgboost_parm['learning_rate'],
            max_depth=xgboost_parm['max_depth'],
            min_child_weight=xgboost_parm['min_child_weight'],
            gamma=xgboost_parm['gamma'],
            reg_alpha=xgboost_parm['reg_alpha'],
            reg_lambda=xgboost_parm['reg_lambda'],
            random_state=RANDOM_STATE
        )

        model.fit(self.x_train, self.t_train)
        train_score, train_error = eval_model(model, self.x_train, self.t_train, 'train')
        val_score, val_error = eval_model(model, self.x_val, self.t_val, 'val')

        log_result(f'MSE for Train: {train_error}', name='XGboost')
        log_result(f'R2-score for Train: {train_score}', name='XGboost')
        log_result(f'MSE for Val: {val_error}', name='XGboost')
        log_result(f'R2-score for Val: {val_score}', name='XGboost')
        param = {
            'n_estimators': xgboost_parm['n_estimators'],
            'learning_rate': xgboost_parm['learning_rate'],
            'max_depth': xgboost_parm['max_depth'],
            'min_child_weight': xgboost_parm['min_child_weight'],
            'gamma': xgboost_parm['gamma'],
            'reg_alpha': xgboost_parm['reg_alpha'],
            'reg_lambda': xgboost_parm['reg_lambda']
        }
        log_result(f'parameters: {str(param)}', name='XGboost')
        log_result('--'*40, name='XGboost')

        return model



def eval_model(model, x, t, name='val'):
    pred = model.predict(x)
    mse_error = mean_squared_error(t, pred)
    r2score = r2_score(t, pred)
    return r2score,mse_error


if __name__ == '__main__':
    x_train, x_val, encode_season, encode_store, t_train, t_val, poly, scaler = PrepareData(TRAIN_VAL).prepare_data()
    # print(x_train.shape, x_val.shape, t_train.shape, t_val.shape) 

    train =Train(x_train, x_val, t_train, t_val)
    model = train.try_xgboost()
    # save_model(model, encode_season, encode_store, poly, scaler, 'XGboost', 'test')

    print("Successful")

