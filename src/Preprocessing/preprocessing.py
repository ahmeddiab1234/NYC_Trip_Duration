

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler, MinMaxScaler, PolynomialFeatures

from data.load_data import load_df, load_x_t, split_data, split_train_val
from Enum.features_enums import FeatureEnum as fenum
from Enum.paths_enums import  PathEnum as penum

from Config.load import load_config
config = load_config()

train_path = penum.TRAIN_PATH.value

drop_outlier = config['preprocessing']['drop_outlier']
calculate_havesine = config['preprocessing']['calculate_haversine']
best_features = config['preprocessing']['best_features']
degree = config['preprocessing']['polynomial']['degree']
include_bias = config['preprocessing']['polynomial']['include_bias']
option = config['preprocessing']['scaling']['option']


class Preprocessing_Pipeline():
    def __init__(self):
        self.poly = None
        self.scaler = None
        self.outlier_limits = {}
        self.label_encoder_store = None
        self.label_encoder_season = None

    def __compute_outlier_limits(self, df: pd.DataFrame):
        limits = {}
        columns = df.select_dtypes(include=np.number).columns
        for col in columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3-q1
            lower = q1 - 1.5*iqr
            upper = q3 + 1.5*iqr
            limits[col] = (lower, upper)
        return limits
    
    def __apply_outlier_limit(self, df: pd.DataFrame):
        for col, (lower, upper) in self.outlier_limits.items():
            if col in df.columns:
                df[col] = df[col].clip(lower, upper)
        return df
    
    def _calucluate_haversine(self, df):
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            lat1,lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2-lat1
            dlon = lon2-lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            return 2*R*np.arcsin(np.sqrt(a)) 

        df[fenum.HAVERSINE_DISTANCE.value] = haversine(
            df[fenum.PICKUP_LATITUDE.value], df[fenum.PICKUP_LONGITUDE.value],
            df[fenum.DROPOFF_LATITUDE.value], df[fenum.DROPOFF_LONGITUDE.value]
        )

        return df


    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()

        if fenum.ID.value in df.columns:
            df.drop(columns=fenum.ID.value, inplace=True, axis=1)
        
        df[fenum.PICKUP_DATETIME.value] = pd.to_datetime(df[fenum.PICKUP_DATETIME.value])
        df[fenum.YEAR.value] = df[fenum.PICKUP_DATETIME.value].dt.year
        df[fenum.MONTH.value] = df[fenum.PICKUP_DATETIME.value].dt.month
        df[fenum.HOUR.value] = df[fenum.PICKUP_DATETIME.value].dt.hour
        df[fenum.DAY_OF_WEEK.value] = df[fenum.PICKUP_DATETIME.value].dt.dayofweek

        def getseason(month):
            if  4 <= month <= 7: return 'Sprint'
            elif  8 <= month <= 10: return 'Summar'
            elif  11 <= month <= 12: return 'Fail'
            else: return 'Winter '

        df[fenum.SEASON.value] = df[fenum.MONTH.value].apply(getseason)

        self.label_encoder_season = LabelEncoder()
        df[fenum.SEASON.value] = self.label_encoder_season.fit_transform(df[fenum.SEASON.value])

        self.label_encoder_store = LabelEncoder()
        df[fenum.STORE_AND_FWD_FLAG.value] = self.label_encoder_store.fit_transform(df[fenum.STORE_AND_FWD_FLAG.value])

        df.drop(columns=fenum.PICKUP_DATETIME, inplace=True)

        if drop_outlier:
            self.outlier_limits = self.__compute_outlier_limits(df)
            df = self.__apply_outlier_limit(df)

        if calculate_havesine:
            df = self._calucluate_haversine(df)

        if best_features:
            df = df[fenum.BEST_FEATURES.value]

        return df, self.label_encoder_season, self.label_encoder_store
    
    def transform(self, df: pd.DataFrame, label_encoder_season, label_encoder_store):
        df = df.copy()
        if fenum.ID.value in df.columns:
            df.drop(columns=fenum.ID.value, inplace=True, axis=1)
                
        df[fenum.PICKUP_DATETIME.value] = pd.to_datetime(df[fenum.PICKUP_DATETIME.value])
        df[fenum.YEAR.value] = df[fenum.PICKUP_DATETIME.value].dt.year
        df[fenum.MONTH.value] = df[fenum.PICKUP_DATETIME.value].dt.month
        df[fenum.HOUR.value] = df[fenum.PICKUP_DATETIME.value].dt.hour
        df[fenum.DAY_OF_WEEK.value] = df[fenum.PICKUP_DATETIME.value].dt.dayofweek

        def getseason(month):
            if  4 <= month <= 7: return 'Sprint'
            elif  8 <= month <= 10: return 'Summar'
            elif  11 <= month <= 12: return 'Fail'
            else: return 'Winter '

        df[fenum.SEASON.value] = df[fenum.MONTH.value].apply(getseason)
        df[fenum.SEASON.value] = label_encoder_season.transform(df[fenum.SEASON.value])
        df[fenum.STORE_AND_FWD_FLAG.value] = label_encoder_store.transform(df[fenum.STORE_AND_FWD_FLAG.value])

        if drop_outlier:
            df = self.__apply_outlier_limit(df)

        if calculate_havesine:
            df = self._calucluate_haversine(df)

        if best_features:
            df = df[fenum.BEST_FEATURES.value]

        return df
    
    def polynomial_feature(self, x, x_val=None):
        self.poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        x = self.poly.fit_transform(x)
        if x_val is not None:
            x_val = self.poly.transform(x_val)
            return self.poly, x, x_val
        
        return self.poly, x

    def scaling(self, x, x_val=None):
        if option == 1:
            self.scaler = MinMaxScaler()
        elif option == 2:
            self.scaler = StandardScaler()
        elif option == 3:
            self.scaler = Normalizer()
        else:
            return None, x, x_val

        x = self.scaler.fit_transform(x)
        if x_val is not None:
            x_val = self.scaler.transform(x_val)
            return self.scaler, x, x_val
        
        return self.scaler, x



if __name__ == '__main__':
    df = load_df(train_path)
    print(df.shape)

    preprcess_pipeline = Preprocessing_Pipeline()
    df,_,_ = preprcess_pipeline.fit_transform(df)

    target = fenum.LOG_TRIP_DURATION.value if fenum.LOG_TRIP_DURATION.value in df.columns else fenum.TRIP_DURATION.value

    t = df[target]
    x = df.drop(columns=[fenum.LOG_TRIP_DURATION.value, fenum.TRIP_DURATION.value], errors='ignore')
    
    poly, x = preprcess_pipeline.polynomial_feature(x, None)
    scaler, x = preprcess_pipeline.scaling(x, None)
    print(t.shape, x.shape)


