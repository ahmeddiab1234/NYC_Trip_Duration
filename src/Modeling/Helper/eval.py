import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import mean_squared_error, r2_score


def eval_model(model, x, t, name='val'):
    pred = model.predict(x)
    mse = mean_squared_error(t, pred)
    r2score = r2_score(t, pred)
    print(f'{name} evaluation :\n MSE error: {mse} \t R2Score: {r2score}')

    return r2score, mse