import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from utilis.helper import  normal_process

def apply_log(feature):
    feature = np.log1p(feature)
    return feature

def label_encoding(data, features):
    le = LabelEncoder()
    for feat in features:
        data[feat] = le.fit_transform(data[feat])


def calc_longitude_latitude(data):
    data['latitude'] = data['dropoff_latitude']-data['pickup_latitude']
    data['longitude'] = data['dropoff_longitude']-data['pickup_longitude']


def pickup_datetime_process(data):
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['month'] = data['pickup_datetime'].dt.month
    data['day'] = data['pickup_datetime'].dt.day
    data['hour'] = data['pickup_datetime'].dt.hour

    def getseason(month):
        if 4<=month<=7:
            return 'Sprint'
        elif 8<=month<=10:
            return 'Summar'
        elif 11<=month<=12:
            return 'Fall'
        else:
            return 'Winter'

    data['season'] = data['pickup_datetime'].dt.month.apply(getseason)
    label_encoding(data,['season'])


def processing_data(x_train,x_val=None, process_option=1,is_monomials_process=False,degree=2):
    return normal_process(x_train, x_val,process_option, is_monomials_process,degree)

