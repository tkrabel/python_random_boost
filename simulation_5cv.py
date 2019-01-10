"""
=================================================================
Simulation Study Comparing Random Boosting with Gradient Boosting
=================================================================

Author: Tobias Krabel

Compares Friedman (2001)'s standard Gradient Boosting framework with
Random Tree Depth Injection. In this simulation, both models are tuned
on the same grid using 5-fold CV.
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

# Homegrown
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Constants
N_SAMPLES = 20000
N_INPUTS = 10
N_COMPONENTS = 20
SIGNAL_TO_NOISE = 1.0
N_NOISE = 5
N_ROUNDS = 15

PARAMS = {
    'learning_rate': [0.1],
    'max_depth': [_ for _ in range(2, 9)],
    'n_estimators': [75, 100, 125, 150, 175]
}

# Set seed
np.random.seed(0)

# Results
df_result = pd.DataFrame(columns=['run', 'model', 'rmse', 'mae', 'time_sec'])

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Fit Random Boost
    print(f'... fit Random Boost')
    start_time = time.time()
    # Little trick: inheritance doesn't seem to work properly
    model = RandomBoostingRegressor()
    rb = GridSearchCV(model, param_grid=PARAMS, cv=5, n_jobs=7)
    rb = rb.fit(X_train, y_train)
    time_rb = time.time() - start_time
    print(f'... took {time_rb} seconds')

    # Fit MART
    print(f'... fit MART')
    start_time = time.time()
    model = GradientBoostingRegressor()
    gb = GridSearchCV(model, param_grid=PARAMS, cv=5, n_jobs=7)
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

# Save to file
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
str_lr = ",".join([str(_) for _ in PARAMS["learning_rate"]])
str_md = ",".join([str(_) for _ in PARAMS["max_depth"]])
str_ne = ",".join([str(_) for _ in PARAMS["n_estimators"]])
df_result.to_csv(f'data/simulation_results/rb-vs-xgb-lr{str_lr}-d{str_md}-nest{str_ne}-5cv-{now}.csv',
                 index=False)
SECS = time.time() - START 
print(f'Simulation took {SECS / N_ROUNDS} seconds per round ({SECS} in total)')