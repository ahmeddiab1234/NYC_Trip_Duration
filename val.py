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
NAME = 'XGboost'

if __name__=='__main__':
    df_val = load_df(VAL_PATH)
    prepare = Preprocessing_Pipeling()
    df = prepare.apply_modify_data(df_val, True, True, True, False)
    df, x,t = load_x_t(df)
    _, x = prepare.scaling(x, None, 2)

    model = load_model(NAME, 'val')

    pred = model.predict(x)
    mse_error = mean_squared_error(t, pred)
    r2score = r2_score(t, pred)


