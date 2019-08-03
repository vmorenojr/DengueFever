import pandas as pd
import numpy as np
from scipy import interpolate
from datetime import datetime

# Get the data

dados = pd.read_csv('Dados/dengue.csv.gz')
capitais = pd.read_csv('Dados/capitais.csv')

# Define day 0 and week numbers for 2000, 2010 and 2018

base = datetime.strptime('2000-01-01', '%Y-%m-%d')
semana_inter = [(datetime.strptime(data, '%Y-%m-%d') - base).days/7 
                    for data in ['2000-01-01', '2010-01-01', '2018-01-01']] 


# Slice the original data and generate week numbers

dados = dados[dados['dt_sintoma'] >= '2000-01-01']
dados['semana'] = [(datetime.strptime(data, '%Y-%m-%d') - base).days/7 for data in dados['dt_sintoma']]
dados['populacao'] = np.nan

# Interpolate population data for each city

for i in capitais.index:
    pop = [capitais.pop_2000[i], capitais.pop_2010[i], capitais.pop_2018[i]]
    poly = interpolate.splrep(semana_inter, pop, k=2)
    new_pop = interpolate.splev(dados[dados['municipio'] == capitais.municipio[i]]['semana'], poly)
    dados.loc[dados['municipio'] == capitais.municipio[i], 'populacao'] = np.around(new_pop, 0)

# Calculate dengue cases per capta

dados['por_habitante'] = dados['ocorrencias']/dados['populacao']
dados.drop(columns='semana', inplace=True)

# Write dataframe to CSV file

dados.to_csv('Dados/dengue_por_habitante.csv.gz', index=False)