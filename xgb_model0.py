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

bh = datasets['Belo Horizonte'].copy()

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
    first_lag_date = start_date - timedelta_lag
    last_lag_date = end_date - timedelta_lag
    lagged = df[(df[dates] >= first_lag_date) &
                (df[dates] <= last_lag_date)][outcome]
    return lagged

# Plot train and test time series
def plot_ts(df_train, df_test, dates, outcome):
    plt.figure(figsize=(10,5))
    plt.xlabel('Time')
    plt.ylabel('Number of reports')
    plt.plot(df_train[dates], df_train[outcome], label='Training data')
    plt.plot(df_test[dates], df_test[outcome], label='Test data')
    plt.legend()
    
# Forecast on test set
def plot_performance(base_df, test_df, predicted_df, dates, predicted, date_from, date_to, title=None):
    plt.figure(figsize=(10,5))
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

# Calculates MAPE given y_true and y_pred
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Generate datasets, run XGBoost and present results
def analysis(df, dates, outcome, start_date, end_date, split_date):
    # Split the time series 
    train, test = split_data(df, 'dt_sintoma', split_date)

    # Plot time series
    plot_ts(df_train=train, df_test=test, dates='dt_sintoma', outcome='ocorrencias')
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
    xgb.plot_importance(reg, height=0.9)
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
                    title='Original and Predicted Data')
    plot_performance(base_df=df, 
                    test_df=test,
                    predicted_df=y_pred,
                    dates=dates, 
                    predicted=outcome, 
                    date_from=split_date,
                    date_to=end_date, 
                    title='Test and Predicted Data')
    plt.show()

    # Computing the error
    print('MSE = ', mean_squared_error(y_true=y_test, y_pred=y_pred))
    print('MAE = ', mean_absolute_error(y_true=y_test, y_pred=y_pred))
    print('MAPE = ', mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred),
          '\n', '\n')
    
# --------------
# Baseline model
# --------------

# Define time series start and end, and split date
end_date = dt.strptime('2019-05-05', '%Y-%m-%d')
start_date = dt.strptime('2010-05-05', '%Y-%m-%d')
split_date = dt.strptime('2018-05-05', '%Y-%m-%d')

analysis(bh, 'dt_sintoma', 'ocorrencias', start_date, end_date, split_date)

# -------------------------
# Model with lagged outcome
# -------------------------

# Create lagged outcome and add as new columns
for lag in range(1,9):
    name = 'lag_' + str(lag)
    lagged_ocorrencias = create_lagged(df=bh, 
                                       dates='dt_sintoma', 
                                       outcome='ocorrencias', 
                                       first_date=start_date,
                                       last_date=end_date, 
                                       timedelta_lag=timedelta(weeks=lag))
    lagged_ocorrencias.index += lag
    bh[name] = lagged_ocorrencias
    
bh = bh[(bh['dt_sintoma']>= start_date) & 
        (bh['dt_sintoma']<= end_date)]

analysis(bh, 'dt_sintoma', 'ocorrencias', start_date, end_date, split_date)

# ------------------------------------------
# Model with lagged outcome for other cities
# ------------------------------------------

bh = datasets['Belo Horizonte'].copy()
bh.set_index('dt_sintoma', drop=False, inplace=True)

# Create lagged outcome and add as new columns
for city in municipios:
    df_city = datasets[city].copy()
    df_city.set_index('dt_sintoma', drop=False, inplace=True)
    for lag in range(1,9):
        name = city +'_' + str(lag)
        lagged_ocorrencias = create_lagged(df=df_city, 
                                           dates='dt_sintoma', 
                                           outcome='ocorrencias', 
                                           first_date=start_date,
                                           last_date=end_date, 
                                           timedelta_lag=timedelta(weeks=lag))
        lagged_ocorrencias.index += timedelta(weeks=lag)
        bh[name] = lagged_ocorrencias
        
bh = bh[(bh['dt_sintoma']>= start_date) & 
        (bh['dt_sintoma']<= end_date)]

analysis(bh, 'dt_sintoma', 'ocorrencias', start_date, end_date, split_date)
print()