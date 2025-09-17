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


RANDOM_STATE = 42
TRAIN_PATH = 'split/train.csv'
VAL_PATH = 'split/val.csv'
TRAIN_VAL = 'split/train_val.csv'

class PrepareData():
    def __init__(self, path=TRAIN_PATH):
        self.df = load_df(path)
        self.preprocess_pipeline = Preprocessing_Pipeling()

    def modify_data_(self, drop_outlier=True, apply_log=True, calculate_haversine=True, best_ten_features=True):
        return self.preprocess_pipeline.apply_modify_data(self.df,drop_outlier,apply_log,calculate_haversine,best_ten_features)

    def polynomial_feature_(self, x, x_val, degree=2, include_bias=True):
        return  self.preprocess_pipeline.polynomial_feature(x, x_val, degree, include_bias)

    def scaling_(self, x, x_val, option=1):
        return self.preprocess_pipeline.scaling(x, x_val, option)
    

class Train():
    def __init__(self,x_train, x_val, t_train, t_val):
        self.x_train=x_train
        self.x_val=x_val
        self.t_train=t_train
        self.t_val=t_val

    def try_linear_regression(self, fit_intercept=True):
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(self.x_train, self.t_train)
        # log_result(f'Training Path', 'LinearRegression')
        # log_result(f'Weights: {model.coef_}')
        # log_result(f'Intercept: {model.intercept_}')

        train_score,train_error = eval_model(model,x_train,t_train, 'train')
        val_score,val_error = eval_model(model,x_val,t_val, 'val')
        # log_result(f'MSE for Train: {train_error}','LinearRegression')
        # log_result(f'R2-score for Train: {train_score}','LinearRegression')
        # log_result(f'MSE for Val: {val_error}','LinearRegression')
        # log_result(f'R2-score for Val: {val_score}','LinearRegression')
        # log_result('--'*40,'LinearRegression')
        return model 


    def try_ridge(self, alpha=1, fit_intercept=True, solver='auto',positive=False):

        # log_result(f'Training Path', 'Ridge')
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver, positive=positive, random_state=RANDOM_STATE)
        model.fit(x_train,t_train)

        train_score,train_mse =eval_model(model,x_train,t_train, 'train')
        val_score, val_mse = eval_model(model,x_val,t_val, 'val')

        # log_result(f'MSE for Train: {train_mse}', name='Ridge')
        # log_result(f'R2-score for Train: {train_score}', name='Ridge')
        # log_result(f'MSE for Val: {val_mse}', name='Ridge')
        # log_result(f'R2-score for Val: {val_score}', name='Ridge')
        # log_result(f'parameters: alpha = {alpha}, fit-intercept = {fit_intercept}', name='Ridge')
        # log_result('--'*40, name='Ridge')

        return model

    def try_neural_network(self, hidden_layers=(16,8), solver='adam', init_lr=0.01, max_iter=500, early_stopping=False, alpha=0.0001):
        # log_result(f'Training Path', 'Neural_Network')
        model = nnr(
            hidden_layer_sizes=hidden_layers,
            activation='identity',
            solver=solver,
            learning_rate='adaptive',
            learning_rate_init=init_lr,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=RANDOM_STATE
        )

        model.fit(x_train, t_train)
        train_score, train_error = eval_model(model, x_train, t_train, 'train')
        val_score, val_error = eval_model(model, x_val, t_val, 'val')

        # log_result(f'Score train: {train_score}', name='Neural_Network')
        # log_result(f'MSE train: {train_error}', name='Neural_Network')
        # log_result(f'Score val: {val_score}', name='Neural_Network')
        # log_result(f'MSE val: {val_error}', name='Neural_Network')

        # current_params = {
        #     'hidden_layers': hidden_layers,
        #     'solver': solver,
        #     'learning_rate_init': init_lr,
        #     'max_iter': max_iter,
        #     'early_stopping': early_stopping,
        #     'alpha': alpha
        # }

        # log_result(f'Parameters: {current_params}', name='Neural_Network')
        # log_result('--'*40, name='Neural_Network')

        return model


    def try_xgboost(self, n_estimators=100, learning_rate=0.05, max_depth=5,min_child_weight=3,gamma=0,reg_alpha=1, reg_lambda=0):
        # log_result(f'Training Path', 'XGboost')

        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=RANDOM_STATE
        )

        model.fit(x_train, t_train)
        train_score, train_error = eval_model(model, x_train, t_train, 'train')
        val_score, val_error = eval_model(model, x_val, t_val, 'val')

        # log_result(f'MSE for Train: {train_error}', name='XGboost')
        # log_result(f'R2-score for Train: {train_score}', name='XGboost')
        # log_result(f'MSE for Val: {val_error}', name='XGboost')
        # log_result(f'R2-score for Val: {val_score}', name='XGboost')
        # param = {
        #     'n_estimators': n_estimators,
        #     'learning_rate': learning_rate,
        #     'max_depth': max_depth,
        #     'min_child_weight': min_child_weight,
        #     'gamma': gamma,
        #     'reg_alpha': reg_alpha,
        #     'reg_lambda': reg_lambda
        # }
        # log_result(f'parameters: {str(param)}', name='XGboost')
        # log_result('--'*40, name='XGboost')

        return model



def eval_model(model, x, t, name='val'):
    pred = model.predict(x)
    mse_error = mean_squared_error(t, pred)
    r2score = r2_score(t, pred)
    return r2score,mse_error


if __name__ == '__main__':
    prepare = PrepareData(TRAIN_PATH)
    df = prepare.modify_data_(True, True, True, False)
    df, x, t = load_x_t(df)
    x_train, x_val, t_train, t_val = split_data(x,t)

    poly, x_train, x_val = prepare.polynomial_feature_(x_train, x_val,2,True)
    scaler, x_train, x_val = prepare.scaling_(x_train, x_val, 2)

    train =Train(x_train, x_val, t_train, t_val)
    model = train.try_linear_regression(True)
    save_model(model, poly, scaler, 'LinearRegression', 'val')


    print("Successful")

