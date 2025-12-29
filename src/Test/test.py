

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd

from .load import load_model
from data.load_data import load_df
from Enum.model_enums import ModelEnum as menum
from Enum.paths_enums import PathEnum as penum
from Enum.features_enums import FeatureEnum as fenum
from Preprocessing.preprocessing import Preprocessing_Pipeline

from sklearn.metrics import r2_score, mean_squared_error
from log.apply_log import log_result

test_path = penum.TEST_PATH.value
name = menum.XGBOOST.value

def test():
    df = load_df(test_path)

    model, encode_season, encode_store, poly, scaler = load_model(name, "test")

    prepare = Preprocessing_Pipeline()
    df = prepare.transform(df, encode_season, encode_store)

    target = fenum.LOG_TRIP_DURATION.value if fenum.LOG_TRIP_DURATION.value in df.columns else fenum.TRIP_DURATION.value

    t = df[target]
    x = df.drop(columns=[fenum.LOG_TRIP_DURATION.value, fenum.TRIP_DURATION.value], errors='ignore')
    t = np.log1p(t)

    x = poly.transform(x)
    x = scaler.transform(x)

    pred = model.predict(x)
    mse_error = mean_squared_error(t, pred)
    r2score = r2_score(t, pred)

    log_result(f'Model Testing : {name}')
    log_result(f'R2Score: {r2_score}, MSE: {mse_error}', name, "test")
    log_result('--'*40, name, "test")
    print(f'r2score: {r2score}, mse: {mse_error}')


if __name__== '__main__':
    test()
    print('Success')


