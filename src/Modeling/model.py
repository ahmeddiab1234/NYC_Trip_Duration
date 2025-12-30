import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from Modeling.Train.xgboost import Train
from Helper.save import save_model
from Helper.prepare import Preparing
from Enum.model_enums import ModelEnum as menum


if __name__ == '__main__':
    prepare = Preparing()
    x_train, x_val, encode_season, encode_store, t_train, t_val, poly, scaler = prepare.prepare_data()
    print(x_train.shape, t_train.shape, x_val.shape, t_val.shape)    

    train = Train(x_train, x_val, t_train, t_val)
    model = train.try_xgboost()
    # eval_model(model, x_train, t_train, 'train')
    # eval_model(model, x_val, t_val, 'val')

    save_model(model, encode_season, encode_store, poly, scaler, menum.XGBOOST.value, 'test')


    print("Successful")







