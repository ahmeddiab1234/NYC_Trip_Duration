"""
load data
prepare data
load pikle file
predict
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Processing.preprocessing import Preprocessing_Pipeling
from utils.helper_fun import *
from sklearn.metrics import r2_score, mean_squared_error


VAL_PATH = 'split/val.csv'
NAME = 'LinearRegression'

if __name__=='__main__':
    df_val = load_df(VAL_PATH)
    prepare = Preprocessing_Pipeling()
    df = prepare.apply_modify_data(df_val, True, True, True, False)
    df, x,t = load_x_t(df)

    model, poly, scaler = load_model(NAME, 'val')
    x = poly.transform(x)
    x = scaler.transform(x)

    pred = model.predict(x)
    mse_error = mean_squared_error(t, pred)
    r2score = r2_score(t, pred)
    print(f'r2score: {r2score}, mse_error: {mse_error}')
    


"""
xgboost regressor val: r2score: 0.997128983245669, mse_error: 0.014643436601412752
Ridge regressor val: r2score: 0.6417990490195127, mse_error: 1.8269809496360412
Linear regression val: r2score: 0.9426316050983026, mse_error: 0.29260381445025296

"""