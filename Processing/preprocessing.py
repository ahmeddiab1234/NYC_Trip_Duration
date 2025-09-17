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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from utils.helper_fun import *


TRAIN_PATH = 'split/train.csv'


class Modify_Data():
    def __init__(self, df:pd.DataFrame):
        self.df = df

    def drop_id(self):
        if 'id' in self.df.columns:
            self.df.drop(columns='id',axis=1, inplace=True)
        return self.df


    def drop_outlier(self):
        columns = self.df.select_dtypes(include=np.number).columns

        for col in columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3-q1
            lower = q1-1.5*iqr
            upper = q3+1.5*iqr
            self.df[col] = self.df[col].clip(lower, upper)
        return self.df


    def apply_log(self):
        self.df['log_trip_duration'] = np.log1p(self.df['trip_duration'])
        return self.df


    def change_datetime(self):
        self.df['pickup_datetime'] = pd.to_datetime(self.df['pickup_datetime'])
        self.df['year'] = self.df['pickup_datetime'].dt.year
        self.df['month'] = self.df['pickup_datetime'].dt.month
        self.df['hour'] = self.df['pickup_datetime'].dt.hour
        self.df['day_of_week'] = self.df['pickup_datetime'].dt.dayofweek

        def getseason(month):
            if 4<=month<=7:
                return 'Sprint'
            elif 8<=month<=10:
                return 'Summar'
            elif 11<=month<=12:
                return 'Fall'
            else:
                return 'Winter'

        self.df['season'] = self.df['pickup_datetime'].dt.month.apply(getseason)
        self.df['season'] = LabelEncoder().fit_transform(self.df['season'])
        self.df.drop(columns='pickup_datetime', axis=1, inplace=True)

        return self.df
    

    def encode_store_fwd_flag(self):
        feat = 'store_and_fwd_flag'
        le = LabelEncoder()
        self.df[feat] = le.fit_transform(self.df[feat])
        return self.df


    def calculate_haversine(self):
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0  
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1,lon1,lat2,lon2])
            dlat = lat2-lat1
            dlon = lon2-lon1

            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2)**2)

            d = 2*R*np.arcsin(np.sqrt(a))
            return d
        
        self.df['haversine_distance'] = haversine(self.df['pickup_latitude'], self.df['pickup_longitude'],self.df['dropoff_latitude'],self.df['dropoff_longitude'])
        return self.df
    

    def best_8_features(self):
        self.df = self.df[['haversine_distance','dropoff_longitude', 'pickup_longitude'
                    ,'month','dropoff_latitude',  
                    'pickup_latitude','season','trip_duration']]

        return self.df



class Preprocessing_Pipeling():
    def __init__(self):
        pass

    def apply_modify_data(self, df:pd.DataFrame,drop_outlier=True, apply_log=True, calculate_haversine=True, best_ten_features=True):

        modify = Modify_Data(df)

        df = modify.drop_id()
        df = modify.change_datetime()
        df = modify.encode_store_fwd_flag()

        if drop_outlier:
            df = modify.drop_outlier()
        if apply_log:
            df = modify.apply_log()
        if calculate_haversine:
            df = modify.calculate_haversine()
        if best_ten_features:
            df = modify.best_8_features()
        return df


    def polynomial_feature(self, x:pd.DataFrame, x_val=None, degree=2, include_bias=True):
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        x = poly.fit_transform(x)

        if x_val is not None:
            x_val = poly.transform(x_val)
            return poly, x, x_val
        return poly, x


    def scaling(self, x:pd.DataFrame, x_val=None, option=1, type='train'):
        if option == 1:
            scaler = MinMaxScaler()
        elif option == 2:
            scaler = StandardScaler()
        elif option == 3:
            scaler = Normalizer()
        else:
            if x_val is not None:
                return None, x,x_val
            else:
                return None, 
        
        x = scaler.fit_transform(x)
        if x_val is not None:
            x_val = scaler.transform(x_val)
            return scaler, x, x_val
        return scaler, x


if __name__=='__main__':
    df = load_df(TRAIN_PATH)
    print(df.shape) # (1000000, 10)

    preprocess_pipeline = Preprocessing_Pipeling()
    df = preprocess_pipeline.apply_modify_data(df, True, False, True, False) 
    
    df, x,t = load_x_t(df)
    print(x.shape, t.shape) # (1000000, 14) (1000000,)
    x_train, x_val, t_train, t_val = split_data(x, t, 0.2) 

    poly, x_train, x_val = preprocess_pipeline.polynomial_feature(x_train, x_val, 2, True)
    scaler, x_train, x_val = preprocess_pipeline.scaling(x_train,x_val, 1)
    # print(x.shape, t.shape) # all data (1000000, 105) (1000000,)
    # print(x.shape, t.shape) # best 7 features (1000000, 36) (1000000,)


    print(x_train.shape, x_val.shape, t_train.shape, t_val.shape) # (800000, 105) (200000, 105) (800000,) (200000,)
    

