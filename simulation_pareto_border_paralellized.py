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
from sklearn.model_selection import train_test_split, GridSearchCV

import pandas as pd 
import numpy as np

import time
import datetime
import itertools
import multiprocessing as mp

# Homegrown --------------------------------------------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred): 
    OUT = np.abs(np.array(y_true) - np.array(y_pred)) / np.abs(np.array(y_true))
    OUT[np.isnan(OUT)] = 0
    return np.mean(OUT)

def all_equal(x, y, idx):
    """
    Compares if dict x contains the same elements as row index idx of np.DF y
    """
    return all(x[i] == y.loc[idx, i] for i in x.keys())

def append_all(x, val):
    """Append val to each element of x (nested list)"""
    for el in x:
        el.append(val)

def copy_all(x):
    """Copy all sublists of x"""
    res = []
    for el in x:
        res.append(el.copy())
    return res

def score_and_time(arg):
    name, func, param_row, best_params, i, idx = arg
    reg = func(n_estimators=param_row.loc[idx, 'n_estimators'], 
               learning_rate=param_row.loc[idx, 'learning_rate'], 
               max_depth=param_row.loc[idx, 'max_depth'])
    start_time = time.time()
    reg = reg.fit(X_train, y_train)
    train_time = time.time() - start_time
    out = pd.DataFrame(
        data={
            'run': [i + 1],
            'model': [name],
            'mae': [mae(y_test, reg.predict(X_test))],
            'rmse': [rmse(y_test, reg.predict(X_test))],
            'time_sec': [train_time],
            'lr': [param_row.loc[idx, 'learning_rate']],
            'd': [param_row.loc[idx, 'max_depth']],
            'm': [param_row.loc[idx, 'n_estimators']],
            'best_cv': [all_equal(best_params, param_row, idx)]
        }
    )
    return out

# Constants --------------------------------------------------------------------
# General
SEED = 1234567

# DGP                    Friedman (2001) Default
N_SAMPLES = 20000
N_INPUTS = 10            # = 10
N_COMPONENTS = 20        # = 20
N_NOISE = 5              # = 0
SIGNAL_TO_NOISE = 1.0
N_ROUNDS = 1

# Tuning settings
CV_FOLDS = 5
N_CORES = 7

MODELS = {
    'RB': RandomBoostingRegressor,
    'MART': GradientBoostingRegressor
}

PARAMS = {
    'learning_rate': [0.1],
    'max_depth': [_ for _ in range(2, 9)],
    'n_estimators': [_ for _ in range(100, 200, 5)]
}

# Create search grid in df
_list = []
for i in itertools.product(*PARAMS.values(), repeat=1):
    _list.append(list(i))

df_grid = pd.DataFrame(data=_list, columns=PARAMS.keys())
n_grid = df_grid.shape[0]

# Set seed
np.random.seed(SEED)

# Results
df_skeleton = pd.DataFrame(columns=['run', 'model', 'mae', 'rmse', 
                                    'time_sec', 'lr', 'd', 'm', 'best_cv'])
df_result = df_skeleton

# Main -------------------------------------------------------------------------
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
        print(f'\n... Model: {name} ---------------------')
        # 1. Find the best model parameters 
        print(f'... Find the best model parameters of {name}')
        model_cv = GridSearchCV(func(), param_grid=PARAMS, cv=CV_FOLDS, 
                                n_jobs=N_CORES)
        model_cv = model_cv.fit(X_train, y_train)
        best_params = model_cv.best_params_

        # 2. Train all models and get test scores
        print(f'... fit {name} again for each tuning parameter constellation')
        
        # Capture accuracy and training time for all points on search grid
        # Create arg list for parallel jobs 
        arg_list = []
        for idx in range(n_grid):
            el = (name, func, df_grid.loc[[idx],:], best_params, i, idx)
            arg_list.append(el)

        # Run Parallel job
        with mp.Pool(processes=7) as pool:
            res = pool.map(score_and_time, arg_list)

        # Consolidate output
        _df_res = pd.concat(res)

        # Add result to main container
        df_result = pd.concat([df_result, _df_res], axis=0, ignore_index=True)

    print('')

# Save to file
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
str_lr = ",".join([str(_) for _ in PARAMS["learning_rate"]])
str_md = ",".join([str(_) for _ in PARAMS["max_depth"]])
str_ne = ",".join([str(_) for _ in PARAMS["n_estimators"]])
#df_result.to_csv(f'data/simulation_results/rb-vs-xgb-accuracy-vs-time-lr{str_lr}-d{str_md}-nest{str_ne}-seed{SEED}-{now}.csv',
#                 index=False)

END = time.time()
print(f'END \nTook {END - START} seconds')
print(df_result)