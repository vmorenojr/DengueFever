import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash_table
import IBGE as ib
from dash.dependencies import Input, Output

import datetime as dt
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Read the data file into a dataframe

dados = pd.read_csv('Dados/dengue_por_habitante.csv.gz')
dados = dados[dados['dt_sintoma'] >= '2006-01-01']

def filter_regiao(value):
   
    traces = []
    
    if type(value) == str:
        value = [value]
    
    for regiao in value:
        print(regiao)
        dados_regiao = dados[dados['regiao'] == regiao]
        dados_regiao.head()
        traces.append([
            go.Scatter(
                x = dados_regiao[dados_regiao['co_municipio'] == i]['dt_sintoma'],
                y = dados_regiao[dados_regiao['co_municipio'] == i]['ocorrencias'],
                text = dados_regiao[dados_regiao['co_municipio'] == i]['municipio'],
                mode = 'lines',
                name = dados_regiao[dados_regiao['co_municipio'] == i]['municipio'].iloc[0]
                ) for i in dados_regiao['co_municipio'].unique()
            ])
    
    return {
            'data': traces,
            'layout': go.Layout(
                template='plotly_white',
                title='Weekly Reports of Dengue Fever in Brazilian State Capitals',            
                xaxis={'title': 'Date'},
                yaxis={'title': 'Weekly Reports'},
                width=1200, height=900,
                hovermode='closest',
                xaxis_rangeslider_visible=True
            )
        }

x = filter_regiao('Sudeste')

y = filter_regiao(['Sudeste', 'Norte'])

print(x)
print(y)

print('final')