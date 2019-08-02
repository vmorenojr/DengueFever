import datetime as dt
import pandas as pd

# Read the data file into a dataframe

dados = pd.read_csv('Dados/dengue_capitais.csv.gz')

# Group the dengue cases by week and city

dados = dados[['dt_diagnostico_sintoma', 'co_municipio_residencia']]
dados.dt_diagnostico_sintoma = pd.to_datetime(dados.dt_diagnostico_sintoma, format = '%Y-%m-%d %H:%M:%S')
dados.set_index(keys='dt_diagnostico_sintoma', inplace=True)
dados = dados.groupby(['co_municipio_residencia']).resample('W').count()
dados.rename_axis(['co_municipio', 'dt_sintoma'], inplace=True)
dados.reset_index(inplace=True)

# Get IBGE city data

municipios = pd.read_csv('Dados/codigos_ibge.csv')
municipios = municipios[['Código Município Completo','Nome_Município']]
municipios['co_municipio'] = (municipios['Código Município Completo']/10).apply(int)

# Add complete IBGE city code and name to dataframe

dados = dados.merge(municipios, left_on='co_municipio', right_on='co_municipio')
dados.drop(columns='co_municipio', inplace = True)
dados.rename(columns={'co_municipio_residencia':'ocorrencias',
                      'Código Município Completo':'co_municipio',
                      'Nome_Município':'municipio'},
              inplace=True)
dados = dados[['dt_sintoma','co_municipio','municipio','ocorrencias']]

# Write dataframe to CSV file

dados.to_csv('Dados/dengue.csv.gz', index=False, date_format = '%Y-%m-%d')