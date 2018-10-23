"""
=================================================================
Simulation Study Comparing Random Boosting with Gradient Boosting
=================================================================

Author: Tobias Krabel

Compares Friedman (2001)'s standard Gradient Boosting framework with
Random Tree Depth Injection.
"""

import random_boost
from random_boost.random_boost import RandomBoostingRegressor, RandomBoostingClassifier

from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd 
import numpy as np

import time

# Homegrown
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

PARAMS = {
    'learning_rate':0.1,
    'max_depth':6,
    'n_estimators':100
}

NOISE = 1
N_SAMPLES = 20000
N_FEATURES = 25
N_ROUNDS = 5

# Results
df_result = pd.DataFrame(columns=['run', 'model', 'rmse', 'mae', 'time_sec'])

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
    start_time = time.time()
    rb = RandomBoostingRegressor(n_estimators=PARAMS['n_estimators'],
                                 learning_rate=PARAMS['learning_rate'],
                                 max_depth=PARAMS['max_depth'])
    rb = rb.fit(X_train, y_train)
    time_rb = time.time() - start_time
    print(f'... took {time_rb} seconds')

    print(f'...fit MART')
    start_time = time.time()
    gb = GradientBoostingRegressor(n_estimators=PARAMS['n_estimators'],
                                   learning_rate=PARAMS['learning_rate'],
                                   max_depth=PARAMS['max_depth'])
    gb = gb.fit(X_train, y_train)
    time_gb = time.time() - start_time
    print(f'... took {time_gb} seconds\n')


    models = [rb, gb]
    df_result = pd.concat([
        df_result,
        pd.DataFrame(
            data={
                'run':([i+1] * 2),
                'model':['rb', 'xgb'],
                'rmse':[rmse(y_test, model.predict(X_test)) for model in models],
                'mae':[mae(y_test, model.predict(X_test)) for model in models],
                'time_sec':[time_rb, time_gb]
            })],
        axis=0,
        ignore_index=True
    )

