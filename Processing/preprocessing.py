"""
- drop id
- remove dublicate
- try to use log or origin trip duration
- change pickup_datetime to datetime and extract time variation
- encoding store_and_fwd_flag to int 
- remove or keep outlier 
- claculate haversine distance

- try to use all data or first 9 with target

- Scaling (minmax scaler, standar scaler, normalize)
- polynomial feature


"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler, MinMaxScaler, Normalizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helper_fun import *

config = load_config()
TRAIN_PATH = config['dataset']['train']

drop_outlier = config['preprocessing']['drop_outlier']
apply_log = config['preprocessing']['apply_log']
calculate_haversine = config['preprocessing']['calculate_haversine']
best_features = config['preprocessing']['best_features']
degree = config['preprocessing']['polynomial']['degree']
include_bias = config['preprocessing']['polynomial']['include_bias']
option = config['preprocessing']['scaling']['option']


class Preprocessing_Pipeling():
    def __init__(self):
        self.label_encoder_store = None
        self.label_encoder_season = None
        self.outlier_limits = {}
        self.poly = None
        self.scaler = None

    def _compute_outlier_limits(self, df):
        limits = {}
        columns = df.select_dtypes(include=np.number).columns
        for col in columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            limits[col] = (lower, upper)
        return limits

    def _apply_outlier_limits(self, df):
        for col, (lower, upper) in self.outlier_limits.items():
            if col in df.columns:
                df[col] = df[col].clip(lower, upper)
        return df

    def _calculate_haversine(self, df):
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            return 2 * R * np.arcsin(np.sqrt(a))
        df['haversine_distance'] = haversine(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )
        return df

    def fit_transform(self, df):
        df = df.copy()

        if 'id' in df.columns:
            df.drop(columns='id', inplace=True)

        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['year'] = df['pickup_datetime'].dt.year
        df['month'] = df['pickup_datetime'].dt.month
        df['hour'] = df['pickup_datetime'].dt.hour
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

        def getseason(month):
            if 4 <= month <= 7: return 'Sprint'
            elif 8 <= month <= 10: return 'Summar'
            elif 11 <= month <= 12: return 'Fall'
            else: return 'Winter'

        df['season'] = df['month'].apply(getseason)

        self.label_encoder_season = LabelEncoder()
        df['season'] = self.label_encoder_season.fit_transform(df['season'])

        self.label_encoder_store = LabelEncoder()
        df['store_and_fwd_flag'] = self.label_encoder_store.fit_transform(df['store_and_fwd_flag'])

        df.drop(columns='pickup_datetime', inplace=True)

        if drop_outlier:
            self.outlier_limits = self._compute_outlier_limits(df)
            df = self._apply_outlier_limits(df)

        if apply_log:
            df['log_trip_duration'] = np.log1p(df['trip_duration'])

        if calculate_haversine:
            df = self._calculate_haversine(df)

        if best_features:
            df = df[['haversine_distance','dropoff_longitude','pickup_longitude',
                    'month','dropoff_latitude','pickup_latitude','season','trip_duration']]

        return df, self.label_encoder_season, self.label_encoder_store

    def transform(self, df,label_encoder_season, label_encoder_store):
        df = df.copy()

        if 'id' in df.columns:
            df.drop(columns='id', inplace=True)

        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['year'] = df['pickup_datetime'].dt.year
        df['month'] = df['pickup_datetime'].dt.month
        df['hour'] = df['pickup_datetime'].dt.hour
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

        def getseason(month):
            if 4 <= month <= 7: return 'Sprint'
            elif 8 <= month <= 10: return 'Summar'
            elif 11 <= month <= 12: return 'Fall'
            else: return 'Winter'

        df['season'] = df['month'].apply(getseason)
        df['season'] = label_encoder_season.transform(df['season'])
        df['store_and_fwd_flag'] = label_encoder_store.transform(df['store_and_fwd_flag'])

        df.drop(columns='pickup_datetime', inplace=True)

        if drop_outlier:
            df = self._apply_outlier_limits(df)

        if apply_log:
            df['log_trip_duration'] = np.log1p(df['trip_duration'])

        if calculate_haversine:
            df = self._calculate_haversine(df)

        if best_features:
            df = df[['haversine_distance','dropoff_longitude','pickup_longitude',
                    'month','dropoff_latitude','pickup_latitude','season','trip_duration']]

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


if __name__=='__main__':
    df = load_df(TRAIN_PATH)
    print(df.shape) # (1000000, 10)

    preprocess_pipeline = Preprocessing_Pipeling()
    df,_,_ = preprocess_pipeline.fit_transform(df)

    target = 'log_trip_duration' if 'log_trip_duration' in df.columns else 'trip_duration'

    t = df[target]
    x = df.drop(columns=['log_trip_duration','trip_duration'], errors='ignore')

    poly, x = preprocess_pipeline.polynomial_feature(x,None)
    print(x.shape, t.shape)
    scaler, x = preprocess_pipeline.scaling(x, None)
    
    print(x.shape, t.shape)


