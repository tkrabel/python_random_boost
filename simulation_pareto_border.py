"""
=================================================================
Simulation Study Comparing Random Boosting with Gradient Boosting
=================================================================

Author: Tobias Krabel

Compares Friedman (2001)'s standard Gradient Boosting framework with
Random Tree Depth Injection. 

In this simulation, we compute the pareto border of the tuple 
(preditcion errro, training time) to investigate whether one procedure 
dominates the other
"""

from random_boost.random_boost import RandomBoostingRegressor, RandomBoostingClassifier
from random_boost.utils import gen_friedman_data

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import pandas as pd 
import numpy as np

import time
import datetime
import itertools

# Homegrown
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Constants              Friedman (2001) Default
N_SAMPLES = 20000
N_INPUTS = 10            # = 10
N_COMPONENTS = 20        # = 20
N_NOISE = 5              # = 0
SIGNAL_TO_NOISE = 1.0
N_ROUNDS = 1

MODELS = {
    'rb': RandomBoostingRegressor,
    'gb': GradientBoostingRegressor
}

PARAMS = {
    'lr': [0.1],
    'd': [_ for _ in range(2, 9)],
    'm': [_ for _ in range(60, 200, 5)]
}

# Create search grid in df
_list = []
for i in itertools.product(*PARAMS.values(), repeat=1):
    _list.append(i)

df_grid = pd.DataFrame(data=_list, columns=PARAMS.keys())
n_grid = df_grid.shape[0]

# Set seed
np.random.seed(0)

# Results
df_skeleton = pd.DataFrame(columns=['run', 'model', 'mae', 'time_sec', 'lr', 'd', 'm'])
df_result = df_skeleton

START = time.time()
for i in range(N_ROUNDS):
    print(f'Round #{i+1} of {N_ROUNDS}')

    # Data
    print('... generate Friedman data')
    X, y = gen_friedman_data(n_samples=N_SAMPLES, 
                             n_inputs=N_INPUTS,
                             n_components=N_COMPONENTS,
                             n_noise=N_NOISE,
                             stn=SIGNAL_TO_NOISE)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for name, func in MODELS.items():
        print(f'... fit {name}')

        # Initialize result buffer
        _df_res = df_skeleton

        # Capture accuracy and training time for all points on search grid
        for idx in range(n_grid):
            param_vals = ' '.join([str(_) for _ in list(df_grid.loc[idx,:])])
            print(f'... ... {idx+1} / {n_grid}: {param_vals}')
            start_time = time.time()
            reg = func(n_estimators=df_grid.loc[idx, 'm'],
                       learning_rate=df_grid.loc[idx, 'lr'],
                       max_depth=df_grid.loc[idx, 'd'])
            reg = reg.fit(X_train, y_train)
            train_time = time.time() - start_time
            test_mae = mae(y_test, reg.predict(X_test))
    
            # Commit to buffer
            _df_res = _df_res.append(
                pd.DataFrame(
                    data={
                        'run': [i + 1],
                        'model': [name],
                        'mae': [test_mae],
                        'time_sec': [train_time],
                        'lr': [df_grid.loc[idx, 'lr']],
                        'd': [df_grid.loc[idx, 'd']],
                        'm': [df_grid.loc[idx, 'm']]
                    }
                ), ignore_index=True
            )
            print(f'... ... > Time: {train_time} seconds. MAE: {test_mae}')
            print('')
        
        # Add result
        df_result = pd.concat([df_result, _df_res], axis=0, ignore_index=True)

# Save to file
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
df_result.to_csv(f'data/simulation_results/rb-vs-xgb-accuracy-vs-time-{now}.csv',
                 index=False)

END = time.time()
print(f'END \nTook {END - START} seconds')