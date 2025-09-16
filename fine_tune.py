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
        log_result(f'Training Path', 'LinearRegression')
        # log_result(f'Weights: {model.coef_}')
        # log_result(f'Intercept: {model.intercept_}')

        train_score,train_error = eval_model(model,x_train,t_train, 'train')
        val_score,val_error = eval_model(model,x_val,t_val, 'val')
        log_result(f'MSE for Train: {train_error}','LinearRegression')
        log_result(f'R2-score for Train: {train_score}','LinearRegression')
        log_result(f'MSE for Val: {val_error}','LinearRegression')
        log_result(f'R2-score for Val: {val_score}','LinearRegression')
        log_result('--'*40,'LinearRegression')
        return model 


    def try_ridge(self):
        best_val_score,best_mse = float('-inf'),float('inf')
        best_param = {'best_alpha':0.1, 'best_fit':True}

        alphas = [0.1, 1, 3, 10]
        fit_intercepts = [True, False]

        for alpha in alphas:
            for fit_inter in fit_intercepts:

                model = Ridge(alpha=alpha, fit_intercept=fit_inter)
                model.fit(x_train,t_train)


                train_score,train_mse =eval_model(model,x_train,t_train, 'train')
                val_score, val_mse = eval_model(model,x_val,t_val, 'val')

                log_result(f'MSE for Train: {train_mse}', name='Ridge')
                log_result(f'R2-score for Train: {train_score}', name='Ridge')
                log_result(f'MSE for Val: {val_mse}', name='Ridge')
                log_result(f'R2-score for Val: {val_score}', name='Ridge')
                log_result(f'parameters: alpha = {alpha}, fit-intercept = {fit_inter}', name='Ridge')

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_mse = val_mse
                    best_param['best_alpha'] = alpha
                    best_param['best_fit'] = fit_inter
                log_result('\n', name='Ridge')
        log_result(f'best parameters: {best_param}', name='Ridge')
        log_result(f'best val score is {best_val_score} with mse error {best_mse}', name='Ridge')


    def try_neural_network(self):
        hidden_layers_list = [(32, 16), (64, 32, 16)]
        solvers = ['adam']
        lr_init_list = [0.001, 0.01]
        max_iterations_list = [500, 1000]
        early_stopping_list = [True, False]

        best_val_score, best_mse = float('-inf'), float('inf')
        best_params = {}

        for layer in hidden_layers_list:
                for lr in lr_init_list:
                    for max_iter in max_iterations_list:
                        for early_stop in early_stopping_list:
                            model = nnr(
                                hidden_layer_sizes=layer,
                                activation='identity',
                                solver='adam',
                                learning_rate='adaptive',
                                learning_rate_init=lr,
                                alpha=0.001,
                                max_iter=max_iter,
                                early_stopping=early_stop,
                                random_state=RANDOM_STATE
                            )

                            model.fit(x_train, t_train)
                            train_score, train_error = eval_model(model, x_train, t_train, 'train')
                            val_score, val_error = eval_model(model, x_val, t_val, 'val')

                            log_result(f'Score train: {train_score}', name='Neural_Network')
                            log_result(f'MSE train: {train_error}', name='Neural_Network')
                            log_result(f'Score val: {val_score}', name='Neural_Network')
                            log_result(f'MSE val: {val_error}', name='Neural_Network')

                            current_params = {
                                'hidden_layers': layer,
                                'solver': 'adam',
                                'learning_rate_init': lr,
                                'max_iter': max_iter,
                                'early_stopping': early_stop
                            }

                            log_result(f'Parameters: {current_params}', name='Neural_Network')
                            log_result('\n', name='Neural_Network')

                            if val_score > best_val_score:
                                best_val_score = val_score
                                best_mse = val_error
                                best_params = current_params.copy()

        log_result(f'Best val score: {best_val_score}, with MSE: {best_mse}', name='Neural_Network')
        log_result(f'Best parameters: {best_params}', name='Neural_Network')



    def try_xgboost(self):
        n_estimators_lst = [100, 300]
        lrs = [0.05, 0.1]
        max_depth_lst = [3, 6]
        param = {'best_n_estimator':500, 'best_lr':0.1, 'best_max_depth':3, 'best_min_child':3, 'best_gamma':0, 'best_reg_alpha':0.1,'best_lambda':1}
        best_val_score, best_mse_error = float('-inf'), float('inf')

        for n_est in n_estimators_lst:
            for lr in lrs:
                for mx_dep in max_depth_lst:
                        model = XGBRegressor(
                            n_estimators=n_est,
                            learning_rate=lr,
                            max_depth=mx_dep,
                            min_child_weight=3,
                            gamma=0,
                            reg_alpha=1,
                            reg_lambda=0,
                            random_state=RANDOM_STATE
                        )

                        model.fit(x_train, t_train)
                        train_score, train_error = eval_model(model, x_train, t_train, 'train')
                        val_score, val_error = eval_model(model, x_val, t_val, 'val')

                        log_result(f'MSE for Train: {train_error}', name='XGboost')
                        log_result(f'R2-score for Train: {train_score}', name='XGboost')
                        log_result(f'MSE for Val: {val_error}', name='XGboost')
                        log_result(f'R2-score for Val: {val_score}', name='XGboost')
                        log_result(f'parameters: {str(param)}', name='XGboost')

                        if val_score > best_val_score:
                            best_val_score = val_score
                            best_mse_error = val_error
                            param['best_n_estimator'],param['best_lr'],param['best_max_depth'],param['best_min_child']  = n_est, lr, mx_dep, 3
                            param['best_gamma'],param['best_reg_alpha'],param['best_lambda'] = 0, 0.1, 1
                        log_result('\n', name='XGboost')


        log_result(f'best score {best_val_score}, with mse {best_mse_error}', name='XGboost')

        log_result(str(param), name='XGboost')



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

    poly_degree = [1,2,3]
    
    for deg in poly_degree:

        x_train, x_val = prepare.polynomial_feature_(x_train, x_val,deg,True)
        _, x_train, x_val = prepare.scaling_(x_train, x_val, 2)

        train = Train(x_train, x_val, t_train, t_val)
        log_result(f'Polynomial Degree {deg}','LinearRegression')
        model = train.try_linear_regression(True)


    print("Successful")

