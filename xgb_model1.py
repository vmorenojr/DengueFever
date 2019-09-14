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

# set params
pd.set_option('display.float_format', lambda x: '%.2f' % x)
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
def analysis(datasets, municipios, target_city, dates, outcome, 
             start_date, end_date, split_date, 
             plt_name_base, lag_weeks=0, range_weeks=8, 
             all_cities=True, show=False):
    
    df_city = datasets[city].copy()
    df_city.set_index(dates, drop=False, inplace=True)

    if all_cities:
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
            outcome=outcome, 
            plt_name=('ts-' + plt_name_base))
    if show:
        plt.show()
    else:
        plt.close()
    
    # Create time series features
    X_train, y_train = create_features(train, dates), train[outcome]
    X_test, y_test = create_features(test, dates), test[outcome]

    # Create and Train XGBoost Model
    reg = xgb.XGBRegressor(n_estimators=1000, 
                           objective='reg:squarederror',
                           seed=123)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50, 
            verbose=False)

    # Feature Importances
    xgb.plot_importance(reg, height=.9, max_num_features=20)
    if show:
        plt.show()
    else:
        plt.close()
    
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
                     plt_name=('or_pred-' + plt_name_base), 
                     title='Original and Predicted Data')
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
                     plt_name=('test_pred-' + plt_name_base),
                     title='Test and Predicted Data')
    if show:
        plt.show()
    else:
        plt.close()
    
    # Computing the error
    # print(results_title)
    # print('MSE = ', mean_squared_error(y_true=y_test, y_pred=y_pred))
    # print('MAE = ', mean_absolute_error(y_true=y_test, y_pred=y_pred))
    # print('MAPE = ', mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred),
    #       '\n', '\n')
    
    MSE = mean_squared_error(y_true=y_test, y_pred=y_pred)
    MAE = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    MAPE = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
    return MSE, MAE, MAPE
    
# --------
# Analysis
# --------

# Define time series start and end, and split date
end_date = dt.strptime('2019-05-05', '%Y-%m-%d')
start_date = dt.strptime('2010-05-05', '%Y-%m-%d')
split_date = dt.strptime('2016-05-05', '%Y-%m-%d')

# Define target city
city = 'Belo Horizonte'

# Define outcome and date features
outcome = 'ocorrencias'
dates = 'dt_sintoma'

# Create dataframe to store results
results = pd.DataFrame(columns=['All cities', 'Lag (weeks)', 'Range (weeks)', 'MSE', 'MAE', 'MAPE'])

# Run analysis and store fit results

for i in [1, 12, 24]:
    for j in [8, 12, 16, 20, 24]:
        MSE, MAE, MAPE = analysis(datasets=datasets, municipios=municipios, target_city=city, 
                                  dates=dates, outcome=outcome,
                                  start_date=start_date, end_date=end_date, split_date=split_date, 
                                  plt_name_base='0', 
                                  lag_weeks=i,
                                  range_weeks=j,
                                  all_cities=True,
                                  show = False)

        results = results.append({'All cities': 'Yes', 
                                'Lag (weeks)': i, 
                                'Range (weeks)': j, 
                                'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE}, ignore_index=True)

        
print(results)
    
# Baseline model
# analysis(datasets=datasets, municipios=municipios, target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the baseline model:',
#          plt_name_base='0', 
#          lag_weeks=1,
#          all_cities=False)

# # 1-week lagged model
# MSE, MAE, MAPE = analysis(datasets=datasets, municipios=['Belo Horizonte'], target_city=city, dates=dates, outcome=outcome,
#                             start_date=start_date, end_date=end_date, split_date=split_date, 
#                             plt_name_base='', 
#                             lag_weeks=1,
#                             range_weeks=12,
#                             all_cities=False)
# results = results.append({'All cities': 'No', 
#                           'Lag (weeks)': i, 
#                           'Range (weeks)': j, 
#                           'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE}, ignore_index=True)
        
# print('Results for the baseline model:')
# print('MSE = ', MSE)
# print('MAE = ', MAE)
# print('MAPE = ', MAPE)
# print(results)
    


# 2-week lagged model
# analysis(datasets=datasets, municipios=['Belo Horizonte'], target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 2-week lagged baseline model:',
#          plt_name_base='2w', 
#          lag_weeks=2,
#          all_cities=True)

# # 1-month lagged model
# analysis(datasets=datasets, municipios=['Belo Horizonte'], target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 2-week lagged baseline model:',
#          plt_name_base='2w', 
#          lag_weeks=4,
#          range_weeks=12,
#          all_cities=True)

# 6-month lagged model
# analysis(datasets=datasets, municipios=['Belo Horizonte'], target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 6-month lagged baseline model:',
#          plt_name_base='6m', 
#          lag_weeks=26,
#          all_cities=True)

# 1-year lagged model
# analysis(datasets=datasets, municipios=['Belo Horizonte'], target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 1-year lagged baseline model:',
#          plt_name_base='1y', 
#          lag_weeks=52,
#          all_cities=True)

# 1-week lagged full model
# analysis(datasets=datasets, municipios=municipios, target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 1-week lagged model with all cities:',
#          plt_name_base='1wcap', 
#          lag_weeks=1,
#          all_cities=True)

# # 2-week lagged full model
# analysis(datasets=datasets, municipios=municipios, target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 2-week lagged model with all cities:',
#          plt_name_base='2wcap', 
#          lag_weeks=2,
#          all_cities=True)

# # 1-month lagged full model
# analysis(datasets=datasets, municipios=municipios, target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 1-month lagged model with all cities:',
#          plt_name_base='1mcap', 
#          lag_weeks=4,
#          all_cities=True)

# # 6-month lagged full model
# analysis(datasets=datasets, municipios=municipios, target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 6-month lagged model with all cities:',
#          plt_name_base='1mcap', 
#          lag_weeks=26,
#          all_cities=True)

# # 1-year lagged full model
# analysis(datasets=datasets, municipios=municipios, target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 1-year lagged model with all cities:',
#          plt_name_base='1mcap', 
#          lag_weeks=52,
#          all_cities=True)

# 1-week lagged full model with range of 12 weeks:
# analysis(datasets=datasets, municipios=municipios, target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 1-week lagged model with with range of 12 weeks and all cities:',
#          plt_name_base='1wcap', 
#          lag_weeks=1,
#          range_weeks=12,
#          all_cities=True)

# # 2-week lagged full model with range of 12 weeks
# analysis(datasets=datasets, municipios=municipios, target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 2-week lagged model with with range of 12 weeks and all cities:',
#          plt_name_base='2wcap', 
#          lag_weeks=2,
#          range_weeks=12,
#          all_cities=True)

# # 1-month lagged full model with range of 12 weeks
# analysis(datasets=datasets, municipios=municipios, target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 1-month lagged model with with range of 12 weeks and all cities:',
#          plt_name_base='1mcap', 
#          lag_weeks=4,
#          range_weeks=12,
#          all_cities=True)

# # 6-month lagged full model with range of 12 weeks
# analysis(datasets=datasets, municipios=municipios, target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 6-month lagged model with with range of 12 weeks and all cities:',
#          plt_name_base='1mcap', 
#          lag_weeks=26,
#          range_weeks=12,
#          all_cities=True)

# # 1-year lagged full model with range of 12 weeks
# analysis(datasets=datasets, municipios=municipios, target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 1-year lagged model with range of 12 weeks and all cities:',
#          plt_name_base='1mcap', 
#          lag_weeks=52,
#          range_weeks=12,
#          all_cities=True)

# # 1-month lagged full model with range of 16 weeks
# analysis(datasets=datasets, municipios=municipios, target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 1-month lagged model with with range of 16 weeks and all cities:',
#          plt_name_base='1mcap', 
#          lag_weeks=4,
#          range_weeks=16,
#          all_cities=True)

# # 1-month lagged full model with range of 24 weeks
# analysis(datasets=datasets, municipios=municipios, target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 1-month lagged model with with range of 24 weeks and all cities:',
#          plt_name_base='1mcap', 
#          lag_weeks=4,
#          range_weeks=24,
#          all_cities=True)

# # 1-month lagged full model with range of 28 weeks
# analysis(datasets=datasets, municipios=municipios, target_city=city, dates=dates, outcome=outcome,
#          start_date=start_date, end_date=end_date, split_date=split_date, 
#          results_title='Results for the 1-month lagged model with with range of 28 weeks and all cities:',
#          plt_name_base='1mcap', 
#          lag_weeks=4,
#          range_weeks=28,
#          all_cities=True)