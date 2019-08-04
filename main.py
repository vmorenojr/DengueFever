import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

import datetime as dt
import pandas as pd

# Function to generate a table in dash from a dataframe

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

# Read the data file into a dataframe

dados = pd.read_csv('Dados/dengue_por_habitante.csv.gz')
dados = dados[dados['dt_sintoma'] >= '2006-01-01']

# Read stationarity results

st = pd.read_csv('Dados/stationarity.csv')
st[['test_statistic','p-value']] = st[['test_statistic','p-value']].apply((lambda x: round(x,5)))
st.columns = ['City', 'Test statistic', 'p-Value', 'Stationary']

# Dash setup

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Generate the web page

app.layout = html.Div([
    
    html.H2(children='Degue Fever Epidemics: Using Distance as a Proxy in Forecasting',
            style={
                'textAlign': 'center'}),
    
    html.H3(children='Lucas Ribeiro & Valter Moreno',
            style={
                'textAlign': 'center'
    }),
    
    html.H4(children='Escola de Matemática Aplicada (EMAp) / FGV',
            style={
                'textAlign': 'center'
    }),
    
    dcc.Markdown(children='''
                 ### Introduction
                 This project is part of the InfoDengue project developed by researchers of Fundação Getúlio
                 Vargas and Fundação Oswaldo Cruz.
                 
                 ### Data
                 The data analyzed in this project was obtained in the InfoDengue database. They consist of 
                 a list of individual cases of dengue fever reported by different instances of the Brazilian 
                 health system in the country's state capitals.
                 
                 ### Exploratory Data Analysis
                 '''),
    
    html.H5(children='Weekly reports of dengue fever in Brazilian state capitals'),
    
    generate_table(dados),
    
    dcc.Graph(
        id='time-series-global',
        figure={
            'data': [
                go.Scatter(
                    x = dados.dt_sintoma.unique(),
                    y = dados.groupby('dt_sintoma')['ocorrencias'].sum(),
                    mode = 'lines',
                    line=dict(color='firebrick'),
                    showlegend = False
            )],
            'layout': go.Layout(
                template='plotly_white',
                title='Weekly Reports of Dengue Fever in Brazil',            
                xaxis={'title': 'Year'},
                yaxis={'title': 'Weekly Reports'},
                hovermode='closest'
            )
        }
    ),
    
    dcc.Graph(
        id='time-series',
        figure={
            'data': [
                go.Scatter(
                    x = dados[dados['co_municipio'] == i]['dt_sintoma'],
                    y = dados[dados['co_municipio'] == i]['ocorrencias'],
                    text = dados[dados['co_municipio'] == i]['municipio'],
                    mode = 'lines',
                    name = dados[dados['co_municipio'] == i]['municipio'].iloc[0],
                    legendgroup = dados[dados['co_municipio'] == i]['regiao'].iloc[0] 
                ) for i in dados['co_municipio'].unique()
            ],
            'layout': go.Layout(
                template='plotly_white',
                title='Weekly Reports of Dengue Fever in Brazilian State Capitals',            
                xaxis={'title': 'Date'},
                yaxis={'title': 'Weekly Reports'},
                width=1200, height=900,
                #margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                #legend={'x': 0, 'y': 1},
                hovermode='closest',
                xaxis_rangeslider_visible=True
            )
        }
    ),
    
    dcc.Graph(
        id='time-series-per-capta',
        figure={
            'data': [
                go.Scatter(
                    x = dados[dados['co_municipio'] == i]['dt_sintoma'],
                    y = dados[dados['co_municipio'] == i]['por_habitante'],
                    text = dados[dados['co_municipio'] == i]['municipio'],
                    mode = 'lines',
                    name = dados[dados['co_municipio'] == i]['municipio'].iloc[0],
                    legendgroup = dados[dados['co_municipio'] == i]['regiao'].iloc[0] 
                ) for i in dados['co_municipio'].unique()
            ],
            'layout': go.Layout(
                template='plotly_white',
                title='Weekly Reports of Dengue Fever per Capta in Brazilian State Capitals',            
                xaxis={'title': 'Date'},
                yaxis={'title': 'Weekly Reports per Capta'},
                width=1200, height=900,
                #margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                #legend={'x': 0, 'y': 1},
                hovermode='closest',
                xaxis_rangeslider_visible=True
            )
        }
    ),
    
    dcc.Markdown(children='''
                #### Stationarity Test
                We used the [Augmented Dickey–Fuller test]
                (https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test)
                to assess the stationarity of each state capital's time series. Time series 
                cross-correlations may be inflated by trend and seasonality. It is important,
                therefore to check the stationarity of the series before calculating cross-correlations. 
                Below, we present the test results for the time series of the Brazilian capitals.
                '''
    ),
    
    generate_table(st),

])

if __name__ == '__main__':
    app.run_server(debug=True)