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
    model, encode_season, encode_store, poly, scaler = load_model(NAME, 'val')

    prepare = Preprocessing_Pipeling()
    df_val = prepare.transform(df_val, encode_season, encode_store, True, True, True, False)
    
    target = 'log_trip_duration' if 'log_trip_duration' in df_val.columns else 'trip_duration'

    t = df_val[target]
    x = df_val.drop(columns=['log_trip_duration','trip_duration'], errors='ignore')
    

    x = poly.transform(x)
    x = scaler.transform(x)

    pred = model.predict(x)
    mse_error = mean_squared_error(t, pred)
    r2score = r2_score(t, pred)
    print(f'r2score: {r2score}, mse_error: {mse_error}')
    


"""
XGboost Validation: r2score: 0.7245229869217307, mse_error: 0.17631537458273602
"""