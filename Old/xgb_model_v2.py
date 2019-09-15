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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
def plot_ts(df_train, df_test, dates, outcome, plt_name):
    plt.figure(figsize=(10,5))
    plt.title('Training and Test Time Series')
    plt.xlabel('Time')
    plt.ylabel('Number of reports')
    plt.plot(df_train[dates], df_train[outcome], label='Training data')
    plt.plot(df_test[dates], df_test[outcome], label='Test data')
    plt.legend()
    #plt.savefig('Plots/' + plt_name + '.png')
    #plt.close(fig)
    
# Forecast on test set
def plot_performance(base_df, test_df, predicted_df, dates, predicted, 
                     date_from, date_to, plt_name, title=None):
    #fig = plt.figure(figsize=(10,5))
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
    #plt.savefig('Plots/' + plt_name + '.png')
    #plt.close(fig)

# Calculates MAPE given y_true and y_pred
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Generate datasets, run XGBoost and present results
def analysis(datasets, target_city, dates, outcome, 
             start_date, end_date, split_date, 
             plt_name_base, all_cities=True):
    
    df_city = datasets[city].copy()
    df_city.set_index(dates, drop=False, inplace=True)

    if all_cities:
        # Create lagged outcome and add as new columns
        for lag in range(1,9):
            name = 'lag_' + str(lag)
            lagged_ocorrencias = create_lagged(df=df_city, 
                                            dates=dates, 
                                            outcome=outcome, 
                                            first_date=start_date,
                                            last_date=end_date, 
                                            timedelta_lag=timedelta(weeks=lag))
            lagged_ocorrencias.index += timedelta(weeks=lag)
            df_city[name] = lagged_ocorrencias
            
        df_city = df_city[(df_city[dates]>= start_date) & 
                (df_city[dates]<= end_date)]

    # Split the time series 
    train, test = split_data(df, dates, split_date)

    # Plot time series
    plot_ts(df_train=train, 
            df_test=test, 
            dates=dates, 
            outcome=outcome, 
            plt_name=('ts-' + plt_name_base))
    plt.show()
    
    # Create time series features
    X_train, y_train = create_features(train, dates), train[outcome]
    X_test, y_test = create_features(test, dates), test[outcome]

    # Create and Train XGBoost Model
    reg = xgb.XGBRegressor(n_estimators=1000, 
                        objective='reg:squarederror')
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50, 
            verbose=False)

    # Feature Importances
    xgb.plot_importance(reg, height=.9, max_num_features=20)
    plt.show()
    
    # Generate forecast
    y_pred = reg.predict(X_test)

    # Plot forecast
    plot_performance(base_df=df, 
                     test_df=test,
                     predicted_df=y_pred,
                     dates=dates, 
                     predicted=outcome, 
                     date_from=start_date,
                     date_to=end_date,
                     plt_name=('or_pred-' + plt_name_base), 
                     title='Original and Predicted Data')
    plt.show()
    
    plot_performance(base_df=df, 
                     test_df=test,
                     predicted_df=y_pred,
                     dates=dates, 
                     predicted=outcome, 
                     date_from=split_date,
                     date_to=end_date, 
                     plt_name=('test_pred-' + plt_name_base),
                     title='Test and Predicted Data')
    plt.show()
    
    # Computing the error
    print('MSE = ', mean_squared_error(y_true=y_test, y_pred=y_pred))
    print('MAE = ', mean_absolute_error(y_true=y_test, y_pred=y_pred))
    print('MAPE = ', mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred),
          '\n', '\n')
    
# -------------------
# Analysis parameters
# -------------------

# Define time series start and end, and split date
end_date = dt.strptime('2019-05-05', '%Y-%m-%d')
start_date = dt.strptime('2010-05-05', '%Y-%m-%d')
split_date = dt.strptime('2018-05-05', '%Y-%m-%d')

# Define target city
city = 'Belo Horizonte'

# Define outcome and date features
outcome = 'ocorrencias'
dates = 'dt_sintoma'

# --------------
# Baseline model
# --------------

df_city = datasets[city].copy()
df_city.set_index(dates, drop=False, inplace=True)

print('Results for baseline model:')

analysis(df_city, dates, outcome, start_date, end_date, split_date, 'baseline')

# ----------------------------------
# Model with a 1-week lagged outcome
# ----------------------------------

df_city = datasets[city].copy()
df_city.set_index(dates, drop=False, inplace=True)

# Create lagged outcome and add as new columns
for lag in range(1,9):
    name = 'lag_' + str(lag)
    lagged_ocorrencias = create_lagged(df=df_city, 
                                       dates=dates, 
                                       outcome=outcome, 
                                       first_date=start_date,
                                       last_date=end_date, 
                                       timedelta_lag=timedelta(weeks=lag))
    lagged_ocorrencias.index += timedelta(weeks=lag)
    df_city[name] = lagged_ocorrencias
    
df_city = df_city[(df_city[dates]>= start_date) & 
        (df_city[dates]<= end_date)]

print('Results for baseline model with 1-week lagged outcome:')

analysis(df_city, dates, outcome, start_date, end_date, split_date, '1w')

# ---------------------------------------------------
# Model with a 1-week lagged outcome for other cities
# ---------------------------------------------------

df_city = datasets[city].copy()
df_city.set_index(dates, drop=False, inplace=True)

# Create lagged outcome and add as new columns
for capital in municipios:
    df_capital = datasets[capital].copy()
    df_capital.set_index(dates, drop=False, inplace=True)
    for lag in range(1,9):
        name = capital +'_' + str(lag)
        lagged_ocorrencias = create_lagged(df=df_capital, 
                                           dates=dates, 
                                           outcome=outcome, 
                                           first_date=start_date,
                                           last_date=end_date, 
                                           timedelta_lag=timedelta(weeks=lag))
        lagged_ocorrencias.index += timedelta(weeks=lag)
        df_city[name] = lagged_ocorrencias
        
df_city = df_city[(df_city[dates]>= start_date) & 
        (df_city[dates]<= end_date)]

print('Results for full model with 1-week lagged outcomes:')

analysis(df_city, dates, outcome, start_date, end_date, split_date, '1wcaps')


# ---------------------------------
# Model with 1-year lagged outcome
# ---------------------------------

df_city = datasets[city].copy()
df_city.set_index(dates, drop=False, inplace=True)

# Create lagged outcome and add as new columns
for lag in range(1,9):
    name = 'lag_' + str(lag)
    lagged_ocorrencias = create_lagged(df=df_city, 
                                       dates=dates, 
                                       outcome=outcome, 
                                       first_date=start_date-timedelta(weeks=52),
                                       last_date=end_date-timedelta(weeks=52), 
                                       timedelta_lag=timedelta(weeks=lag))
    lagged_ocorrencias.index += timedelta(weeks=lag)
    df_city[name] = lagged_ocorrencias
    
df_city = df_city[(df_city[dates]>= start_date) & 
        (df_city[dates]<= end_date)]

print('Results for baseline model with 1-y lagged outcomes:')

analysis(df_city, dates, outcome, start_date, end_date, split_date, '1y')

# --------------------------------------------------
# Model with 1-year lagged outcome for other cities
# --------------------------------------------------

df_city = datasets[city].copy()
df_city.set_index(dates, drop=False, inplace=True)

# Create lagged outcome and add as new columns
for capital in municipios:
    df_capital = datasets[capital].copy()
    df_capital.set_index(dates, drop=False, inplace=True)
    for lag in range(1,9):
        name = capital +'_' + str(lag)
        lagged_ocorrencias = create_lagged(df=df_capital, 
                                           dates=dates, 
                                           outcome=outcome, 
                                           first_date=start_date-timedelta(weeks=52),
                                           last_date=end_date-timedelta(weeks=52), 
                                           timedelta_lag=timedelta(weeks=lag))
        lagged_ocorrencias.index += timedelta(weeks=lag)
        df_city[name] = lagged_ocorrencias
        
df_city = df_city[(df_city[dates]>= start_date) & 
        (df_city[dates]<= end_date)]

print('Results for full model with 1-year lagged outcomes:')

analysis(df_city, dates, outcome, start_date, end_date, split_date, '1ycaps')

# ----------------------------------------------------
# Model with a 6-month lagged outcome for other cities
# ----------------------------------------------------

df_city = datasets[city].copy()
df_city.set_index(dates, drop=False, inplace=True)

# Create lagged outcome and add as new columns
for capital in municipios:
    df_capital = datasets[capital].copy()
    df_capital.set_index(dates, drop=False, inplace=True)
    for lag in range(1,9):
        name = capital +'_' + str(lag)
        lagged_ocorrencias = create_lagged(df=df_capital, 
                                           dates=dates, 
                                           outcome=outcome, 
                                           first_date=start_date-timedelta(weeks=26),
                                           last_date=end_date-timedelta(weeks=26), 
                                           timedelta_lag=timedelta(weeks=lag))
        lagged_ocorrencias.index += timedelta(weeks=lag)
        df_city[name] = lagged_ocorrencias
        
df_city = df_city[(df_city[dates]>= start_date) & 
        (df_city[dates]<= end_date)]

print('Results for full model with 6-month lagged outcomes:')

analysis(df_city, dates, outcome, start_date, end_date, split_date, '6mcaps')

# ----------------------------------------------------
# Model with a 3-month lagged outcome for other cities
# ----------------------------------------------------

df_city = datasets[city].copy()
df_city.set_index(dates, drop=False, inplace=True)

# Create lagged outcome and add as new columns
for capital in municipios:
    df_capital = datasets[capital].copy()
    df_capital.set_index(dates, drop=False, inplace=True)
    for lag in range(1,9):
        name = capital +'_' + str(lag)
        lagged_ocorrencias = create_lagged(df=df_capital, 
                                           dates=dates, 
                                           outcome=outcome, 
                                           first_date=start_date-timedelta(weeks=12),
                                           last_date=end_date-timedelta(weeks=12), 
                                           timedelta_lag=timedelta(weeks=lag))
        lagged_ocorrencias.index += timedelta(weeks=lag)
        df_city[name] = lagged_ocorrencias
        
df_city = df_city[(df_city[dates]>= start_date) & 
        (df_city[dates]<= end_date)]

print('Results for full model with 3-month lagged outcomes:')

analysis(df_city, dates, outcome, start_date, end_date, split_date, '3mcaps')

# ----------------------------------------------------
# Model with a 1-month lagged outcome for other cities
# ----------------------------------------------------

df_city = datasets[city].copy()
df_city.set_index(dates, drop=False, inplace=True)

# Create lagged outcome and add as new columns
for capital in municipios:
    df_capital = datasets[capital].copy()
    df_capital.set_index(dates, drop=False, inplace=True)
    for lag in range(1,9):
        name = capital +'_' + str(lag)
        lagged_ocorrencias = create_lagged(df=df_capital, 
                                           dates=dates, 
                                           outcome=outcome, 
                                           first_date=start_date-timedelta(weeks=4),
                                           last_date=end_date-timedelta(weeks=4), 
                                           timedelta_lag=timedelta(weeks=lag))
        lagged_ocorrencias.index += timedelta(weeks=lag)
        df_city[name] = lagged_ocorrencias
        
df_city = df_city[(df_city[dates]>= start_date) & 
        (df_city[dates]<= end_date)]

print('Results for full model with 1-month lagged outcomes:')

analysis(df_city, dates, outcome, start_date, end_date, split_date, '1mcaps')

# ---------------------------------------------------
# Model with a 15-day lagged outcome for other cities
# ---------------------------------------------------

df_city = datasets[city].copy()
df_city.set_index(dates, drop=False, inplace=True)

# Create lagged outcome and add as new columns
for capital in municipios:
    df_capital = datasets[capital].copy()
    df_capital.set_index(dates, drop=False, inplace=True)
    for lag in range(1,9):
        name = capital +'_' + str(lag)
        lagged_ocorrencias = create_lagged(df=df_capital, 
                                           dates=dates, 
                                           outcome=outcome, 
                                           first_date=start_date-timedelta(weeks=2),
                                           last_date=end_date-timedelta(weeks=2), 
                                           timedelta_lag=timedelta(weeks=lag))
        lagged_ocorrencias.index += timedelta(weeks=lag)
        df_city[name] = lagged_ocorrencias
        
df_city = df_city[(df_city[dates]>= start_date) & 
        (df_city[dates]<= end_date)]

print('Results for full model with 2-week lagged outcomes:')

analysis(df_city, dates, outcome, start_date, end_date, split_date, '2wcaps1')

# ---------------------------------------------------------
# Model with a 15-day lagged outcome and earlier split date
# ---------------------------------------------------------

split_date = dt.strptime('2017-05-05', '%Y-%m-%d')

df_city = datasets[city].copy()
df_city.set_index(dates, drop=False, inplace=True)

# Create lagged outcome and add as new columns
for capital in municipios:
    df_capital = datasets[capital].copy()
    df_capital.set_index(dates, drop=False, inplace=True)
    for lag in range(1,9):
        name = capital +'_' + str(lag)
        lagged_ocorrencias = create_lagged(df=df_capital, 
                                           dates=dates, 
                                           outcome=outcome, 
                                           first_date=start_date-timedelta(weeks=2),
                                           last_date=end_date-timedelta(weeks=2), 
                                           timedelta_lag=timedelta(weeks=lag))
        lagged_ocorrencias.index += timedelta(weeks=lag)
        df_city[name] = lagged_ocorrencias
        
df_city = df_city[(df_city[dates]>= start_date) & 
        (df_city[dates]<= end_date)]

print('Results for full model with 2-weeks lagged outcomes and split date 05-05-2017:')

analysis(df_city, dates, outcome, start_date, end_date, split_date, '2wcaps2')

# --------------------------------------------------------------
# Model with a 15-day lagged outcome and even earlier split date
# --------------------------------------------------------------

split_date = dt.strptime('2016-05-05', '%Y-%m-%d')

df_city = datasets[city].copy()
df_city.set_index(dates, drop=False, inplace=True)

# Create lagged outcome and add as new columns
for capital in municipios:
    df_capital = datasets[capital].copy()
    df_capital.set_index(dates, drop=False, inplace=True)
    for lag in range(1,9):
        name = capital +'_' + str(lag)
        lagged_ocorrencias = create_lagged(df=df_capital, 
                                           dates=dates, 
                                           outcome=outcome, 
                                           first_date=start_date-timedelta(weeks=2),
                                           last_date=end_date-timedelta(weeks=2), 
                                           timedelta_lag=timedelta(weeks=lag))
        lagged_ocorrencias.index += timedelta(weeks=lag)
        df_city[name] = lagged_ocorrencias
        
df_city = df_city[(df_city[dates]>= start_date) & 
        (df_city[dates]<= end_date)]

print('Results for full model with 2-weeks lagged outcomes and split date 05-05-2016:')

analysis(df_city, dates, outcome, start_date, end_date, split_date, '2wcaps3')


# --------------------------------------------------------------
# Model with a 1-year lagged outcome and even earlier split date
# --------------------------------------------------------------

split_date = dt.strptime('2016-05-05', '%Y-%m-%d')

df_city = datasets[city].copy()
df_city.set_index(dates, drop=False, inplace=True)

# Create lagged outcome and add as new columns
for capital in municipios:
    df_capital = datasets[capital].copy()
    df_capital.set_index(dates, drop=False, inplace=True)
    for lag in range(1,9):
        name = capital +'_' + str(lag)
        lagged_ocorrencias = create_lagged(df=df_capital, 
                                           dates=dates, 
                                           outcome=outcome, 
                                           first_date=start_date-timedelta(weeks=52),
                                           last_date=end_date-timedelta(weeks=52), 
                                           timedelta_lag=timedelta(weeks=lag))
        lagged_ocorrencias.index += timedelta(weeks=lag)
        df_city[name] = lagged_ocorrencias
        
df_city = df_city[(df_city[dates]>= start_date) & 
        (df_city[dates]<= end_date)]

print('Results for full model with 1-year lagged outcomes and split date 05-05-2016:')

analysis(df_city, dates, outcome, start_date, end_date, split_date, '1ycaps3')


# --------------------------------------------------------------
# Model with a 15-day lagged outcome and split date '05-05-2015'
# --------------------------------------------------------------

start_date = dt.strptime('2008-05-05', '%Y-%m-%d')
split_date = dt.strptime('2015-05-05', '%Y-%m-%d')

df_city = datasets[city].copy()
df_city.set_index(dates, drop=False, inplace=True)

# Create lagged outcome and add as new columns
for capital in municipios:
    df_capital = datasets[capital].copy()
    df_capital.set_index(dates, drop=False, inplace=True)
    for lag in range(1,9):
        name = capital +'_' + str(lag)
        lagged_ocorrencias = create_lagged(df=df_capital, 
                                           dates=dates, 
                                           outcome=outcome, 
                                           first_date=start_date-timedelta(weeks=2),
                                           last_date=end_date-timedelta(weeks=2), 
                                           timedelta_lag=timedelta(weeks=lag))
        lagged_ocorrencias.index += timedelta(weeks=lag)
        df_city[name] = lagged_ocorrencias
        
df_city = df_city[(df_city[dates]>= start_date) & 
        (df_city[dates]<= end_date)]

print('Results for full model with 2-weeks lagged outcomes and split date 05-05-2015:')

analysis(df_city, dates, outcome, start_date, end_date, split_date, '2wcaps3')


