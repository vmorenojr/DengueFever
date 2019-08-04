import datetime as dt
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Read the data file into a dataframe

dados = pd.read_csv('Dados/dengue_por_habitante.csv.gz')
dados = dados[dados['dt_sintoma'] >= '2006-01-01'].set_index('dt_sintoma')

# Run Augmented Dickey-Fuller unit root test of time series stationarity

stationary = []

for city in dados.municipio.unique():
    serie = dados[dados.municipio == city]['ocorrencias']
    adf = adfuller(serie)
    stationary.append([city, adf[0], adf[1], adf[1]<.01])

stationary = pd.DataFrame(stationary, columns=['municipio','test_statistic','p-value','stationary'])

stationary.to_csv('Dados/stationarity.csv', index=False)