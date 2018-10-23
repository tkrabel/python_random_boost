from random_boost.random_boost import RandomBoostingRegressor

from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd 
import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

params = {
    'learning_rate':0.1,
    'max_depth':6,
    'n_estimators':100
}

NOISE = 1
N_SAMPLES = 20000
N_FEATURES = 25
N_ROUNDS = 5

# Results
df_result = pd.DataFrame(columns=['run', 'model', 'rmse', 'mae'])

for i in range(N_ROUNDS):
    print(f'Round #{i+1}')
    # Data
    X_train, y_train = make_regression(n_samples=N_SAMPLES, 
                                       n_features=N_FEATURES,
                                       noise=NOISE)
    X_test, y_test = make_regression(n_samples=N_SAMPLES, 
                                     n_features=N_FEATURES,
                                     noise=NOISE)

    print(f'...fit Random Boost')
    rb = RandomBoostingRegressor(n_estimators=params['n_estimators'],
                                 learning_rate=params['learning_rate'],
                                 max_depth=params['max_depth'])
    rb = rb.fit(X_train, y_train)

    print(f'...fit MART')
    gb = GradientBoostingRegressor(n_estimators=params['n_estimators'],
                                   learning_rate=params['learning_rate'],
                                   max_depth=params['max_depth'])
    gb = gb.fit(X_train, y_train)

    print('\n')

    models = [rb, gb]
    df_result = pd.concat([
        df_result,
        pd.DataFrame(
            data={
                'run':([i+1] * 2),
                'model':['rb', 'xgb'],
                'rmse':[rmse(y_test, model.predict(X_test)) for model in models],
                'mae':[mae(y_test, model.predict(X_test)) for model in models]
            })],
        axis=0,
        ignore_index=True
    )

