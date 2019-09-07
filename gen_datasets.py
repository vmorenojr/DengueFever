# standard library
from datetime import datetime as dt
from datetime import timedelta

# pydata stack
import pandas as pd

# set params
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Read the main data file into a dataframe
dados = pd.read_csv('Dados/dengue_por_habitante.csv.gz')
dados = dados[dados['dt_sintoma'] >= '2006-01-01']

inicio = dados['dt_sintoma'].min()
fim = dados['dt_sintoma'].max()

municipios = dados['municipio'].unique()

# Create dataframe with data aggregated by month
dados_mes = dados[dados['dt_sintoma'] >= '2012-08-01'].drop('populacao', axis=1)
dados_mes['dt_sintoma']=dados_mes['dt_sintoma'].apply(lambda x: x[:7])

ocorrencias_mes=dados_mes.groupby(['co_municipio','dt_sintoma'])[['ocorrencias','por_habitante']].sum()
ocorrencias_mes.reset_index(inplace=True)

dados_mes.drop_duplicates(['co_municipio','dt_sintoma'], inplace=True)
dados_mes = dados_mes.merge(ocorrencias_mes, 
                left_on=['co_municipio','dt_sintoma'],
                right_on=['co_municipio','dt_sintoma'])
dados_mes.drop(['ocorrencias_x', 'por_habitante_x'], axis=1, inplace=True)
dados_mes.rename(columns={'ocorrencias_y':'ocorrencias', 'por_habitante_y':'por_habitante'}, inplace=True)

# Read the distances bewteen state capitals into a dataframe
dist = pd.read_csv('Dados/distancias.csv')

# Slice dataframe by date
def slice_date(df, date1=inicio, date2=fim):
    return df[(df['dt_sintoma'] >= date1) & 
              (df['dt_sintoma'] <= date2)]
    
# Calculate window
def window(start, years=3):
    start_date = dt.strptime(start, '%Y-%m-%d')
    end_date = start_date - timedelta(days=365*years, hours=0, minutes=0)
    return end_date.strftime('%Y-%m-%d')

# Get data for target date and window
def window_data(df, target_date, window_length):
    most_recent = dt.strptime(target_date, '%Y-%m-%d') - timedelta(days=7)
    most_recent = most_recent.strftime('%Y-%m-%d')
    earliest_date = window(most_recent, window_length)
    return slice_date(df, earliest_date, most_recent)

# Get distances between cities
def get_distances(df, city):
    return df[df['municipio'] == city]

# Create dataset
def gen_dataset(df, target_city, 
                target_start_date, 
                target_years_training = 5, 
                window_years = 3):
    
    target_dates = slice_date(df['dt_sintoma'].to_frame(), 
                            window(target_start_date, target_years_training),
                            target_start_date)
    target_dates = target_dates['dt_sintoma'].unique()
    target_dates = target_dates[:(len(target_dates)-1)]

    ocorr_alvo = df.loc[(df['municipio'] == target_city) &
                        (df['dt_sintoma'] == target_start_date), 'ocorrencias'].values[0]   
    
    por_hab_alvo = df.loc[(df['municipio'] == target_city) &
                          (df['dt_sintoma'] == target_start_date), 'por_habitante'].values[0] 
    
    dados_modelo = window_data(df, target_start_date, window_years)
    dados_modelo.insert(0, 'capital', target_city)
    dados_modelo.insert(1, 'data_alvo', target_start_date)
    dados_modelo.insert(2, 'ocorrencias_alvo', ocorr_alvo)
    dados_modelo.insert(3, 'por_habitante_alvo', por_hab_alvo)
    
    lags = pd.to_datetime(dados_modelo['dt_sintoma'], format='%Y-%m-%d')
    lags = lags.apply(lambda x: (dt.strptime(target_start_date, '%Y-%m-%d')-x).days/7)
    dados_modelo.insert(6, 'lag', lags)
 
    dados_modelo.insert(dados_modelo.shape[1], 'distancia', 0)  
  
    for date in target_dates:
        ocorr_alvo = df.loc[(df['municipio'] == target_city) &
                            (df['dt_sintoma'] == date), 'ocorrencias'].values[0]   
    
        por_hab_alvo = df.loc[(df['municipio'] == target_city) &
                              (df['dt_sintoma'] == date), 'por_habitante'].values[0] 
     
        dados_window = window_data(df, date, window_years)
        dados_window.insert(0, 'capital', target_city)
        dados_window.insert(1, 'data_alvo', date)
        dados_window.insert(2, 'ocorrencias_alvo', ocorr_alvo)
        dados_window.insert(3, 'por_habitante_alvo', por_hab_alvo)
        dados_window.insert(dados_window.shape[1], 'distancia',0)
        
        lags = pd.to_datetime(dados_window['dt_sintoma'], format='%Y-%m-%d')
        lags = lags.apply(lambda x: (dt.strptime(date, '%Y-%m-%d')-x).days/7)
        dados_window.insert(6, 'lag', lags)
        
        dados_modelo = dados_modelo.append(dados_window)

    for city in dist['municipio']:
        valor = dist.loc[dist['municipio']==target_city, city].values[0]
        dados_modelo.loc[dados_modelo['municipio']==city, 'distancia'] = valor/100

    return dados_modelo

# Create and save datasets
for city in municipios:
    fim = dados[dados['municipio']==city]['dt_sintoma'].max()
    new_dataset = gen_dataset(dados[['municipio', 'dt_sintoma', 'ocorrencias', 
                                     'por_habitante']], 
                              target_city=city,
                              target_start_date=fim,
                              target_years_training = 5, 
                              window_years = 3)
    file_path = 'Dados/Datasets/' + city + '.csv.gz'
    new_dataset.to_csv(file_path, index=False)

