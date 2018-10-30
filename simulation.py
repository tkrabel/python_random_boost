"""
=================================================================
Simulation Study Comparing Random Boosting with Gradient Boosting
=================================================================

Author: Tobias Krabel

Compares Friedman (2001)'s standard Gradient Boosting framework with
Random Tree Depth Injection.
"""

from random_boost.random_boost import RandomBoostingRegressor, RandomBoostingClassifier
from random_boost.utils import gen_friedman_data

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import pandas as pd 
import numpy as np

import time

# Homegrown
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Constants
N_SAMPLES = 10000
N_INPUTS = 10
N_COMPONENTS = 20
SIGNAL_TO_NOISE = 1.0
N_ROUNDS = 5

PARAMS = {
    'learning_rate':0.1,
    'max_depth':6,
    'n_estimators':100
}

# Set seed
np.random.seed(0)

# Results
df_result = pd.DataFrame(columns=['run', 'model', 'rmse', 'mae', 'time_sec'])

for i in range(N_ROUNDS):
    print(f'Round #{i+1}')

    # Data
    print('... generate Friedman data')
    X, y = gen_friedman_data(n_samples=N_SAMPLES, 
                             n_inputs=N_INPUTS,
                             n_components=N_COMPONENTS,
                             stn=SIGNAL_TO_NOISE)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Fit Random Boost
    print(f'... fit Random Boost')
    start_time = time.time()
    rb = RandomBoostingRegressor(n_estimators=PARAMS['n_estimators'],
                                 learning_rate=PARAMS['learning_rate'],
                                 max_depth=PARAMS['max_depth'])
    rb = rb.fit(X_train, y_train)
    time_rb = time.time() - start_time
    print(f'... took {time_rb} seconds')

    # Fit MART
    print(f'... fit MART')
    start_time = time.time()
    gb = GradientBoostingRegressor(n_estimators=PARAMS['n_estimators'],
                                   learning_rate=PARAMS['learning_rate'],
                                   max_depth=PARAMS['max_depth'])
    gb = gb.fit(X_train, y_train)
    time_gb = time.time() - start_time
    print(f'... took {time_gb} seconds\n')

    # Add Results to Container
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

