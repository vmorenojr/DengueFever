import pandas as pd
from statsmodels.tsa.stattools import adfuller
from itertools import permutations

# Function to calculate cross-correlation with lag

def crosscorr(serie1, serie2, lag=0):
    return serie1.corr(serie2.shift(lag))

# Read the data file into a dataframe

dados = pd.read_csv('Dados/dengue_por_habitante.csv.gz')
dados = dados[dados['dt_sintoma'] >= '2006-01-01'].set_index('dt_sintoma')

# Run Augmented Dickey-Fuller unit root test of time series stationarity

cities = dados.municipio.unique()
stationary = []

for city in cities:
    serie = dados[dados.municipio == city]['ocorrencias']
    adf = adfuller(serie)
    stationary.append([city, adf[0], adf[1], adf[1]<.01])

stationary = pd.DataFrame(stationary, columns=['municipio','test_statistic','p-value','stationary'])
stationary.to_csv('Dados/stationarity.csv', index=False)

cities = [(city,city) for city in dados.municipio.unique()]
cities.extend(list(permutations(dados.municipio.unique(), 2)))
cities = [[city1, city2] for (city1, city2) in cities]

all_corrs = pd.DataFrame(cities, columns=['municipio1', 'municipio2'])\
    .sort_values(by=['municipio1', 'municipio2'])

for lag in range(0, 13):
    
    corrs = pd.DataFrame(index=dados.municipio.unique(),
                         columns=dados.municipio.unique())
    
    lag_col = 'lag_'+str(lag)
    
    all_corrs[lag_col] = ''
        
    for [city1, city2] in cities:
        corrs.loc[city1, city2] = crosscorr(dados[dados.municipio == city1]['ocorrencias'],
                                            dados[dados.municipio == city2]['ocorrencias'],
                                            lag=lag)
        all_corrs.loc[(all_corrs['municipio1'] == city1) & 
                      (all_corrs['municipio2'] == city2), lag_col] = corrs.loc[city1, city2]
    
    corrs.to_csv('Dados/corrs_{}.csv'.format(str(lag)))

all_corrs.to_csv('Dados/all_corrs.csv', index=False)