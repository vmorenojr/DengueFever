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
from datetime import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# set params
pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(12345)

# -------------
# Load the data
# -------------

dados = pd.read_csv('Dados/Dengue_por_habitante.csv.gz')

datasets = {}
municipios = dados['municipio'].unique()

for city in municipios:
    df = dados[(dados['dt_sintoma'] >= '2006-01-01') & (dados['municipio'] == city)].copy()
    df.drop(['co_municipio', 'municipio', 'estado', 'UF', 'regiao', 'latitude', 'longitude',
            'populacao'], axis=1, inplace=True)
    df.dt_sintoma = pd.to_datetime(df.dt_sintoma, format = '%Y-%m-%d %H:%M:%S')
    datasets[city] = df

# -------------------
# Auxiliary functions
# -------------------

# Split time series
def split_data(df, dates, split_date):
    return df[df[dates] <= split_date].copy(), \
           df[df[dates] >  split_date].copy()

# Create and append time series features
def create_features(df, dates):
    df_new = df.copy()
    df_new['year'] = df[dates].dt.year
    df_new['quarter'] = df[dates].dt.quarter
    df_new['month'] = df[dates].dt.month
    df_new['weekofyear'] = df[dates].dt.weekofyear
    df_new['dayofyear'] = df[dates].dt.dayofyear
    df_new['dayofmonth'] = df[dates].dt.day
    df_new.drop([dates, 'ocorrencias', 'por_habitante'], 
                axis=1, inplace=True)
    return df_new

# Create lagged outcome
def create_lagged(df, dates, outcome, first_date, last_date, timedelta_lag):
    first_lag_date = first_date - timedelta_lag
    last_lag_date = last_date - timedelta_lag
    lagged = df[(df[dates] >= first_lag_date) &
                (df[dates] <= last_lag_date)][outcome]
    return lagged

# Plot train and test time series
def plot_ts(df_train, df_test, dates, outcome):#, plt_name):
    fig=plt.figure(figsize=(10,5))
    plt.title('Training and Test Time Series')
    plt.xlabel('Time')
    plt.ylabel('Number of reports')
    plt.plot(df_train[dates], df_train[outcome], label='Training data')
    plt.plot(df_test[dates], df_test[outcome], label='Test data')
    plt.legend()
    return fig
    
# Forecast on test set
def plot_performance(base_df, test_df, predicted_df, dates, predicted, 
                     date_from, date_to, title=None):
    fig = plt.figure(figsize=(10,5))
    if title == None:
        plt.title('From {0} to {1}'.format(date_from, date_to))
    else:
        plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Number of reports')
    plt.plot(base_df[dates], base_df[predicted], label='Data')
    plt.plot(test_df[dates], predicted_df, label='Prediction')
    plt.legend()
    plt.xlim(left=date_from, right=date_to)
    return fig

# Calculates MAPE given y_true and y_pred
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Generate datasets, run XGBoost and present results
def analysis(datasets, municipios, target_city, dates, outcome, 
             start_date, end_date, split_date, 
             plt_name_base, pars, lag_weeks=0, range_weeks=8,
             lagged=True, date_features=True, show=False):
    
    df_city = datasets[city].copy()
    df_city.set_index(dates, drop=False, inplace=True)

    if lagged:
        # Create lagged outcome and add as new columns
        for capital in municipios:
            df_capital = datasets[capital].copy()
            df_capital.set_index(dates, drop=False, inplace=True)
            for lag in range(1, range_weeks+1):
                name = capital +'_' + str(lag)
                lagged_ocorrencias = create_lagged(df=df_capital, 
                                                   dates=dates, 
                                                   outcome=outcome, 
                                                   first_date=start_date-timedelta(weeks=lag_weeks-1),
                                                   last_date=end_date-timedelta(weeks=lag_weeks-1), 
                                                   timedelta_lag=timedelta(weeks=lag))
                lagged_ocorrencias.index += timedelta(weeks=lag_weeks-1) + timedelta(weeks=lag)
                df_city[name] = lagged_ocorrencias
                
    df_city = df_city[(df_city[dates]>= start_date) & 
            (df_city[dates]<= end_date)]

    # Split the time series 
    train, test = split_data(df_city, dates, split_date)

    # Plot time series
    plot_ts(df_train=train, 
            df_test=test, 
            dates=dates, 
            outcome=outcome)
    plt.savefig('Plots/ts-' + plt_name_base + '.png')
    if show:
        plt.show()
    else:
        plt.close()
    
    # Create time series features
    if date_features:
        X_train, y_train = create_features(train, dates), train[outcome]
        X_test, y_test = create_features(test, dates), test[outcome]
    else:
        X_train, y_train = train.drop([dates,'ocorrencias', 'por_habitante'], axis=1), train[outcome]
        X_test, y_test = test.drop([dates,'ocorrencias', 'por_habitante'], axis=1), test[outcome]

        # Create and Train XGBoost Model
    
    # Fit the model
    reg = xgb.XGBRegressor(params=pars, 
                           objective='reg:squarederror',
                           seed=123)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50, 
            verbose=False)

    # Plot and save feature importances
    plt.rcParams['figure.figsize'] = (15, 10)
    xgb.plot_importance(reg, height=.9, max_num_features=30)
    plt.savefig('Plots/imp-' + plt_name_base + '.png')
    if show:
        plt.show()
    else:
        plt.close()
    
    reg_importance = pd.DataFrame(reg.feature_importances_, columns=['Importance'])*1000
    reg_importance['Features'] = X_train.columns
    
    if lagged: 
        city_lag = reg_importance.Features.str.split('_', expand=True)
        reg_importance = reg_importance.merge(city_lag, left_index=True, right_index=True)\
                                       .drop('Features', axis=1)
        reg_importance.columns = ['Importance', 'City', 'Lag']
    
    reg_importance.to_csv('XGB_fit/importances.csv', index=False)
    
    # Generate forecast
    y_pred = reg.predict(X_test)

    # Plot forecast
    plot_performance(base_df=df_city, 
                     test_df=test,
                     predicted_df=y_pred,
                     dates=dates, 
                     predicted=outcome, 
                     date_from=start_date,
                     date_to=end_date, 
                     title='Original and Predicted Data')
    #plt.savefig('Plots/tspred-' + plt_name_base + '.png')
    if show:
        plt.show()
    else:
        plt.close()
    
    plot_performance(base_df=test, 
                     test_df=test,
                     predicted_df=y_pred,
                     dates=dates, 
                     predicted=outcome, 
                     date_from=split_date,
                     date_to=end_date, 
                     title='Test and Predicted Data')
    plt.savefig('Plots/pred-' + plt_name_base + '.png')
    if show:
        plt.show()
    else:
        plt.close()
    
    RMSE = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    MAE = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    MAPE = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
    return RMSE, MAE, MAPE

# Function for validation and test of model
def training(datasets, municipios, target_city, dates, outcome, 
             start_date, end_date, split_date,
             n_estimators, learning_rate, min_child_weight,
             max_depth, gamma, subsample, colsample_bytree,
             lag_weeks=0, range_weeks=8):
    
    df_city = datasets[city].copy()
    df_city.set_index(dates, drop=False, inplace=True)

    for capital in municipios:
        df_capital = datasets[capital].copy()
        df_capital.set_index(dates, drop=False, inplace=True)
        for lag in range(1, range_weeks+1):
            name = capital +'_' + str(lag)
            lagged_ocorrencias = create_lagged(df=df_capital, 
                                               dates=dates, 
                                               outcome=outcome, 
                                               first_date=start_date-timedelta(weeks=lag_weeks-1),
                                               last_date=end_date-timedelta(weeks=lag_weeks-1), 
                                               timedelta_lag=timedelta(weeks=lag))
            lagged_ocorrencias.index += timedelta(weeks=lag_weeks-1) + timedelta(weeks=lag)
            df_city[name] = lagged_ocorrencias
                
    df_city = df_city[(df_city[dates]>= start_date) & 
            (df_city[dates]<= end_date)]

    # Split the time series
    train, test = split_data(df_city, dates, split_date)

    X_train, y_train = train.drop([dates,'ocorrencias', 'por_habitante'], axis=1), train[outcome]
    X_test, y_test = test.drop([dates,'ocorrencias', 'por_habitante'], axis=1), test[outcome]

    # Create and Train XGBoost Model
    reg = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                           min_child_weight=min_child_weight, max_depth=max_depth, 
                           gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree,
                           objective='reg:squarederror', seed=123)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50, 
            verbose=False)

    # Generate forecast
    y_pred = reg.predict(X_test)

    return np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    
# --------
# Analysis
# --------

# Define time series start and end, and split date
end_date = dt.strptime('2019-05-05', '%Y-%m-%d')
start_date = dt.strptime('2010-05-05', '%Y-%m-%d')
split_date = dt.strptime('2018-05-05', '%Y-%m-%d')

# Define target city
city = 'Belo Horizonte'

# Define outcome and date features
outcome = 'ocorrencias'
dates = 'dt_sintoma'

# --------
# Baseline
# --------

# Run analysis and store fit results
RMSE, MAE, MAPE = analysis(datasets=datasets, municipios=['Belo Horizonte'], target_city=city, 
                           dates=dates, outcome=outcome,
                           start_date=start_date, end_date=end_date, split_date=split_date, 
                           plt_name_base='base',
                           pars={'n_estimators': 1000}, 
                           lag_weeks=1,
                           range_weeks=12,
                           lagged=False,
                           date_features=True,
                           show = False)
results = pd.DataFrame({'RMSE': RMSE, 'MAE': MAE, 'MAPE': MAPE}, index=[0])
results.to_csv('XGB_fit/baseline.csv', index=False)
print(results)

# ------------------
# Baseline with lags
# ------------------

# Create dataframe to store results
results = pd.DataFrame(columns=['All cities', 'Lag (weeks)', 'Range (weeks)', 'RMSE', 'MAE', 'MAPE'])

# Run analysis and store fit results
for i in [4, 12, 26, 52]:
    for j in [8, 12, 26, 52]:
        plt_base = 'base-l' + str(i) + 'r' + str(j)# + 'caps'
        RMSE, MAE, MAPE = analysis(datasets=datasets, municipios=['Belo Horizonte'], target_city=city, 
                                  dates=dates, outcome=outcome,
                                  start_date=start_date, end_date=end_date, split_date=split_date, 
                                  plt_name_base=plt_base, 
                                  pars={'n_estimators': 1000},
                                  lag_weeks=i,
                                  range_weeks=j,
                                  lagged=True,
                                  date_features=True,
                                  show = False)
        results = results.append({'All cities': 'No', 
                                'Lag (weeks)': i, 
                                'Range (weeks)': j, 
                                'RMSE': RMSE, 'MAE': MAE, 'MAPE': MAPE}, ignore_index=True)
results.to_csv('XGB_fit/baseline-lags.csv', index=False)
print(results)

# -----------------------
# Model with all capitals
# -----------------------

# Create dataframe to store results
results = pd.DataFrame(columns=['All cities', 'Lag (weeks)', 'Range (weeks)', 'RMSE', 'MAE', 'MAPE'])

# Run analysis and store fit results
for i in [4, 12, 26, 52]:
    for j in [8, 12, 26, 52]:
        plt_base = 'all-l' + str(i) + 'r' + str(j)# + 'caps'
        RMSE, MAE, MAPE = analysis(datasets=datasets, municipios=municipios, target_city=city, 
                                  dates=dates, outcome=outcome,
                                  start_date=start_date, end_date=end_date, split_date=split_date, 
                                  plt_name_base=plt_base, 
                                  pars={'n_estimators': 1000},
                                  lag_weeks=i,
                                  range_weeks=j,
                                  lagged=True,
                                  date_features=True,
                                  show = False)
        results = results.append({'All cities': 'Yes', 
                                'Lag (weeks)': i, 
                                'Range (weeks)': j, 
                                'RMSE': RMSE, 'MAE': MAE, 'MAPE': MAPE}, ignore_index=True)
results.to_csv('XGB_fit/allcaps.csv', index=False)
print(results)

# ----------------------------------------------------
# Model with all capitals and no time-related features
# ----------------------------------------------------

# Create dataframe to store results
results = pd.DataFrame(columns=['All cities', 'Lag (weeks)', 'Range (weeks)', 'RMSE', 'MAE', 'MAPE'])

# Run analysis and store fit results
for i in [4]:
    for j in [8, 12, 26, 52]:
        plt_base = 'all-no-time-l' + str(i) + 'r' + str(j)# + 'caps'
        RMSE, MAE, MAPE = analysis(datasets=datasets, municipios=municipios, target_city=city, 
                                  dates=dates, outcome=outcome,
                                  start_date=start_date, end_date=end_date, split_date=split_date, 
                                  plt_name_base=plt_base, 
                                  pars={'n_estimators': 1000},
                                  lag_weeks=i,
                                  range_weeks=j,
                                  lagged=True,
                                  date_features=False,
                                  show = False)
        results = results.append({'All cities': 'Yes', 
                                'Lag (weeks)': i, 
                                'Range (weeks)': j, 
                                'RMSE': RMSE, 'MAE': MAE, 'MAPE': MAPE}, ignore_index=True)
results.to_csv('XGB_fit/allcapsnotime.csv', index=False)
print(results)

# -------------------------
# Tuning of hyperparameters
# -------------------------

n_estimators = [500]
learning_rate = [.275]
min_child_weight = [5]
max_depth = [5]
gamma = [0]
subsample = [1]
colsample_bytree=[1]

results = pd.DataFrame(columns=['n_estimators', 'learning_rate', 'min_child_weight', 
                                'max_depth', 'gamma', 'subsample', 'colsample_bytree', 
                                'RMSE'])

for est in n_estimators:
    for learn in learning_rate:
        for min_child in min_child_weight:
            for max_dpth in max_depth:
                for gma in gamma:
                    for sub in subsample:
                        for col in colsample_bytree:
                            RMSE = training(datasets=datasets, municipios=municipios, target_city=city,
                                            dates=dates, outcome=outcome,
                                            start_date=start_date, 
                                            end_date=(end_date - timedelta(weeks=12)), 
                                            split_date=(end_date - timedelta(weeks=24)), 
                                            lag_weeks=4, range_weeks=26,
                                            n_estimators=est,
                                            learning_rate=learn,
                                            min_child_weight=min_child,
                                            max_depth=max_dpth, 
                                            gamma=gma, 
                                            subsample=sub,
                                            colsample_bytree=col)
                                            
                            results = results.append({'n_estimators': est, 'learning_rate': learn,
                                                    'min_child_weight': min_child, 'max_depth': max_dpth, 
                                                    'gamma':gma, 'subsample': sub, 'colsample_bytree': col,
                                                    'RMSE': RMSE}, 
                                                    ignore_index=True)
                            
results.to_csv('XGB_fit/tuning-learning.csv', index=False)
print(results[results.RMSE == min(results.RMSE)])

# -------------------------
# Testing
# -------------------------

results = []
end = start_date + timedelta(weeks = 36)

while end <= end_date:
    results.append(training(datasets=datasets, municipios=municipios, target_city=city,
                            dates=dates, outcome=outcome,
                            start_date=start_date, 
                            end_date=end, 
                            split_date=(end - timedelta(weeks=12)), 
                            lag_weeks=4, range_weeks=26,
                            n_estimators=500,
                            learning_rate=.275,
                            min_child_weight=5,
                            max_depth=5, 
                            gamma=0, 
                            subsample=1,
                            colsample_bytree=1))
    end += timedelta(weeks=12)

print('Mean RMSE = ', np.mean(results))

# Predicting the outcome for the last year
RMSE, MAE, MAPE = analysis(datasets=datasets, municipios=municipios, target_city=city, 
                           dates=dates, outcome=outcome,
                           start_date=start_date, end_date=end_date, split_date=split_date, 
                           plt_name_base='final-no-time-l4r26', 
                           pars={'n_estimators': 500,
                                 'learning_rate': .275,
                                 'min_child_weight': 5,
                                 'max_depth': 5,
                                 'gamma': 0,
                                 'subsample': 1,
                                 'colsample_bytree': 1},
                           lag_weeks=4, range_weeks=26,
                           lagged=True, date_features=False, show = False)

print('RMSE: ', RMSE, 'MAE: ', MAE, 'MAPE: ', MAPE)


