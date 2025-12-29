import pickle

def load_model(name="xgboost", type="val"):
    if type=="val":
        filename = fr'val_pkl/{name}.pkl'
    else:
        filename = fr'{name}.pkl'

    with open(filename, 'rb') as f:
        loaded_dict = pickle.load(f)

    return loaded_dict["model"], loaded_dict["encode_season"], loaded_dict["encode_store"], loaded_dict["poly"], loaded_dict["scaler"]


