
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd

from data.load_data import load_df, load_x_t, split_train_val
from Config.load import load_config
from Enum.features_enums import FeatureEnum as fenum    
from Enum.paths_enums import PathEnum as penum
from Preprocessing.preprocessing import Preprocessing_Pipeline


config = load_config()
train_val_path = penum.TRAIN_VAL_PATH.value
apply_log_train = config['preprocessing']['training']['apply_log']



class Preparing():
    def __init__(self, path=train_val_path):
        self.df = load_df(path)
        self.train, self.val = split_train_val(self.df)
        self.preprocess_pipeline = Preprocessing_Pipeline()

    def prepare_data(self):
        self.train, encode_season, encode_store = self.preprocess_pipeline.fit_transform(self.train)
        self.val = self.preprocess_pipeline.transform(self.val, encode_season, encode_store)
        
        t_train = self.train[fenum.TRIP_DURATION.value]
        t_val = self.val[fenum.TRIP_DURATION.value]

        if apply_log_train:
            t_train = np.log1p(t_train)
            t_val = np.log1p(t_val)

        x_train = self.train.drop(columns=[fenum.TRIP_DURATION.value])
        x_val = self.val.drop(columns=[fenum.TRIP_DURATION.value])

        poly, x_train, x_val = self.preprocess_pipeline.polynomial_feature(x_train, x_val)
        scaler, x_train, x_val = self.preprocess_pipeline.scaling(x_train,x_val)


        return x_train, x_val, encode_season, encode_store, t_train, t_val, poly, scaler

