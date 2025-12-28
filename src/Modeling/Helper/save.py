import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle

def save_model(model, encode_season, encode_store, poly, scaler, name="Xgboost", tyep='val'):
    model_dict = {
        "model": model,
        "encode_season": encode_season,
        "encode_store": encode_store,
        "poly": poly,
        "scaler": scaler
    }

    if type=='val':
        filename = fr'val_pkl/{name}.pkl'
    else:
        filename = fr'{name}.pkl'

    with open(filename, 'wb') as file:
        pickle.dump(model_dict, file)
