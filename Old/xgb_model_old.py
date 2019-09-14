# -------------------
# XGBoost instalation
# -------------------
'''
    To install xgboost: https://www.building-skynet.com/2018/08/03/xgboost-part-un-installation/

    Using this magic website by researchers at UC Irvine where they have put plenty of libraries there. 
    I actually remember using their website in my early Python days to install some libraries 
    (pre Anaconda days!)

    Go to https://www.lfd.uci.edu/~gohlke/pythonlibs/#xgboost and download the correct version 
    of the .whl file for your system. Pay attention to which version of Python you are using 
    and whether you are on a 32 bit or 64 bit machine (if you have only one Python on your machine,
    open Command Prompt and type python –version)

    Open the terminal, cd to the directory where you downloaded the file in and type:
    $ pip install xgboost-0.72-cp36-cp36m-win_amd64.whl (I don`t think I have to remind you to 
                                                        use the file name of your downloaded file!)

    This should be it! Open your code editor and type import xgboost as xgb to see if it works.
'''

# ----------------
# Import libraries
# ----------------

import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# -------------
# Load the data
# -------------

bh = pd.read_csv('Dados/Datasets/Belo Horizonte.csv.gz')

# -------------------
# Auxiliary functions
# -------------------

# Generate training and test datasets
def split(df, outcome, seed=12345):
    y = outcome + '_alvo'
    X, y = df[['lag', outcome, 'distancia']], df[y]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=.25,
                                                        random_state=seed)
    return X_train, X_test, y_train, y_test

# Get number of estimators
def get_tree(X, y, params_dict, folds=5, num_boost_round=500, early_stopping_rounds=25):
    DM_data = xgb.DMatrix(data=X, label=y)
    
    cv_results = xgb.cv(dtrain=DM_data, 
                        params=params_dict,
                        nfold=folds,
                        num_boost_round=num_boost_round,
                        early_stopping_rounds=early_stopping_rounds,
                        metrics='rmse',
                        as_pandas=True,
                        seed=12345)
    
    return cv_results

# Get optimal parameters
def get_hyper(X, y, params_dict, random=True, iterations=100, folds=5):
    gbm = xgb.XGBRegressor()
    
    if random:
        search_mse = RandomizedSearchCV(estimator=gbm,
                                        param_distributions=params_dict,
                                        n_iter=iterations,
                                        scoring='neg_mean_squared_error',
                                        cv=folds,
                                        verbose=1)
    else:    
        search_mse = GridSearchCV(estimator=gbm,
                                  param_grid=params_dict,
                                  scoring='neg_mean_squared_error',
                                  cv=folds,
                                  verbose=1)
    search_mse.fit(X, y)
    return search_mse.best_params_, np.sqrt(np.abs(search_mse.best_score_))

# Test the model
def test(Xtrain, Xtest, ytrain, ytest, params, seed):
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror',
                              params=params,
                              seed=seed)
    xg_reg.fit(Xtrain, ytrain)
    preds = xg_reg.predict(Xtest)
    
    rmse = np.sqrt(mean_squared_error(ytest, preds))
    preds_pd = pd.Series(preds)
    
    return rmse, preds_pd
  
# ------------------------------------------
# Training and validation of hyperparameters
# ------------------------------------------

# Split the dataset
X_train, X_test, y_train, y_test = split(bh, outcome='por_habitante')

X_train.to_csv('Dados/xtrain_bh.csv.gz')
X_test.to_csv('Dados/xtest_bh.csv.gz')
y_train.to_csv('Dados/ytrain_bh.csv.gz')
y_test.to_csv('Dados/ytest_bh.csv.gz')

# Step1: Fix learning rate and number of estimators for tuning tree-based parameters
params = {'learning_rate': .1,
          'min_child_weight': 1,          
          'max_depth': 6,
          'gamma': 0,
          'subsample': 1}

best_estimators = get_tree(X_train, y_train, params_dict=params,
                            num_boost_round=1000, early_stopping_rounds=50)
print('Optimal number of estimators for number of cases in Belo Horizonte:')
print(best_estimators)

plt.plot(best_estimators.index, best_estimators['test-rmse-mean'], label='linear')
plt.show()

best_estimators.reset_index(inplace=True)
best_estimators.rename(columns={'index':'Trees', 'test-rmse-mean': 'Mean RMSE'}, 
                       inplace=True)
best_estimators.to_csv('Dados/xgboost_estimators_1.csv')

# Step 2: Tune max_depth and min_child_weight
params = {'learning_rate': [.1],
          'min_child_weight': [.1, .5, 1, 2, 5],          
          'max_depth': [4, 6, 8, 10],
          'n_estimators': [25],
          'gamma': [0],
          'subsample': [1]}

best_pars, best_rmse = get_hyper(X_train, y_train, random=False, iterations=5, params_dict=params)
print('Best RMSE for number of cases in Belo Horizonte:')
print(best_pars)
print(best_rmse)

# Initial results
# {'subsample': 1, 'n_estimators': 25, 'min_child_weight': 2, 'max_depth': 4, 'learning_rate': 0.1, 'gamma': 0}
# RMSE: 115.55052038968748

# Refining the parameters
params = {'learning_rate': [.1],
          'min_child_weight': [1.5, 2, 2.5],          
          'max_depth': [3, 4, 5],
          'n_estimators': [25],
          'gamma': [0],
          'subsample': [1]}

best_pars, best_rmse = get_hyper(X_train, y_train, random=False, iterations=5, params_dict=params)
print('Best RMSE for number of cases in Belo Horizonte:')
print(best_pars)
print(best_rmse)

# Refined results
# {'subsample': 1, 'n_estimators': 25, 'min_child_weight': 2.5, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0}
# RMSE: 115.52004363204088

# Step 3: Setting the gamma parameter
params = {'learning_rate': [.1],
          'min_child_weight': [2.5],          
          'max_depth': [5],
          'n_estimators': [25],
          'gamma': [0, .1, .25, .5, 1, 2],
          'subsample': [1]}

best_pars, best_rmse = get_hyper(X_train, y_train, random=False, iterations=5, params_dict=params)
print('Best RMSE for number of cases in Belo Horizonte:')
print(best_pars)
print(best_rmse)

# Optimal gamma
# {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 2.5, 'n_estimators': 25, 'subsample': 1}
# RMSE: 115.52004363204088

# Step 4: Setting the subsample and colsample_bytree parameters
params = {'learning_rate': [.1],
          'min_child_weight': [2.5],          
          'max_depth': [5],
          'n_estimators': [25],
          'gamma': [0],
          'subsample': [.3, .6, 1],
          'colsample_bytree': [.3, .6, 1]}

best_pars, best_rmse = get_hyper(X_train, y_train, random=False, iterations=5, params_dict=params)
print('Best RMSE for number of cases in Belo Horizonte:')
print(best_pars)
print(best_rmse)

# Results
# {'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 2.5, 'n_estimators': 25, 'subsample': 1}
# 115.52004363204088

# Step5: Lower the learning rate while increasing the number of estimators
params = {'learning_rate': .01,
          'min_child_weight': 2.5,          
          'max_depth': 5,
          'gamma': 0,
          'subsample': 1}

best_estimators = get_tree(X_train, y_train, params_dict=params,
                            num_boost_round=5000, early_stopping_rounds=50)
print('Optimal number of estimators for number of cases in Belo Horizonte:')
print(best_estimators)

plt.plot(best_estimators.index, best_estimators['test-rmse-mean'], label='linear')
plt.show()

best_estimators.reset_index(inplace=True)
best_estimators.rename(columns={'index':'Trees', 'test-rmse-mean': 'Mean RMSE'}, 
                       inplace=True)
best_estimators.to_csv('Dados/xgboost_estimators_01.csv')

# -----------------
# Testing the model
# -----------------

params = {'learning_rate': .01,
          'n_estimators': 250,
          'min_child_weight': 2.5,          
          'max_depth': 5,
          'gamma': 0,
          'subsample': 1}

rmse_test, predicted = test(X_train, X_test, y_train, y_test, 
                            params=params, 
                            seed=12345)
print('RMSE for the test dataset: % 5.2f' %(rmse_test))

df_predicted = y_test.to_frame().reset_index()
df_predicted['por_habitante_pred'] = predicted
df_predicted = df_predicted.merge(X_test.reset_index(),
                                  left_index=True,
                                  right_index=True)
df_predicted.drop('index_y', axis=1, inplace=True)
df_predicted.rename(columns={'index_x':'index'}, inplace=True)

df_predicted.to_csv('Dados/xgboost_pred.csv.gz')
