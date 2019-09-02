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
#dados.por_habitante = round(100000*dados.por_habitante,2)

# Read stationarity results

st = pd.read_csv('Dados/stationarity.csv')
#st[['test_statistic','p-value']] = st[['test_statistic','p-value']].apply((lambda x: round(x,5)))
st.columns = ['City', 'Test statistic', 'p-Value', 'Stationary']

# Read cross-correlations and prepared dataframe

corrs = pd.read_csv('Dados/all_corrs.csv')
# corrs[['lag_0', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8']] = \
#     corrs[['lag_0', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8']] \
#         .apply((lambda x: round(x,2)))
corrs.columns = ['municipio1', 'municipio2', 0, 1, 2, 3, 4, 5, 6, 7, 8]
corrs = pd.melt(corrs, id_vars = ['municipio1', 'municipio2'], var_name='lag', value_name='correlation')

# Read the data about state capitals into a dataframe

capitais = pd.read_csv('Dados/capitais.csv')

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

    dash_table.DataTable(
        id='datatable-begin',
        columns=[{'name': i, 'id': i, 'deletable': True} for i in dados.columns],
        data=dados.to_dict('records'),
        fixed_rows={'headers': True, 'data': 0},
        style_cell={'width': '150px'},
        filter_action='native'
    ),
    html.Div(id='filter1'),
    
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

    dash_table.DataTable(
        id='st',
        columns=[{'name': i, 'id': i} for i in st.columns],
        data=st.to_dict('records'),
        fixed_rows={'headers': True, 'data': 0},
        style_cell={'width': '40px', 'maxHeight': '5'},
    ),
    
    # Cross-correlation chart:
    
    dcc.Graph(
        id='cross-corrs',
        figure={
            'data': [
                go.Scatter3d(
                    x = corrs[(corrs['municipio1'] == 'Recife') & 
                              (corrs['municipio2'] == i)]['lag'],
                    y = corrs[(corrs['municipio1'] == 'Recife') & 
                              (corrs['municipio2'] == i)]['municipio2'],
                    z = corrs[(corrs['municipio1'] == 'Recife') & 
                              (corrs['municipio2'] == i)]['correlation'],
                    mode = 'lines',
                    surfaceaxis = -1
                ) for i in corrs[(corrs['municipio2'].isin( 
                                  capitais[capitais['regiao'] == 'Nordeste']['municipio'].to_list())) \
                                  & (corrs['municipio2'] != 'Recife')] \
                                ['municipio2'].unique()
            ],
            'layout': go.Layout(
                template = 'plotly_white',
                title = 'Cross-correlations for Dengue Fever Time Series',            
                scene = dict(
                    xaxis_title = 'Time lag (weeks)',
                    yaxis_title = 'City',
                    zaxis_title = 'Correlation'),
                showlegend=False,
                hovermode='closest',
                width = 800, height = 800#,
                #margin = dict(r = 20, b = 10, l = 10, t = 10)
            )
        }
    ),    

    dcc.Markdown(children=
             """#### Monthly/Trimester/Yearly table of reports by region
             """),

    html.Div(
        id = 'container-col-select',
        children = dcc.RadioItems(
        id = 'regions-data',
        options = [{'label':i, 'value':i} for i in ib.regioes.keys()],
        labelStyle={'display':'inline-block'},
        value = 'Norte'
            ),
    ),

    html.Div(
        id = 'container-period-select',
        children = dcc.RadioItems(
            id = 'period-filter',
            options = [{'label':'Trimester','value':'3M'},{'label':'Yearly','value':'Y'},{'label':'Monthlty','value':'M'}],
            value = 'M',
            labelStyle={'display': 'inline-block'}
        )

    ),


    dash_table.DataTable(
        id='datatable-regions',
        columns=[{'name': i, 'id': i} for i in ['dt_sintoma','ocorrencias','populacao','por_habitante']],
        data=dados.to_dict('records'),
        fixed_rows={ 'headers': True, 'data': 0 },
        style_cell={'width': '150px'}
    ),
    html.Div(id='datatable-filter-container')

])



@app.callback(Output('datatable-regions','data'), [Input('regions-data',"value"), Input('period-filter',"value")])
def filter_table(value1,value2):
    if value1 is None:
        dff = dados
    else:
        dff = ib.group_by(dados[dados['regiao']==value1],value2,value1).reset_index()
        dff['por_habitante'] = round(100000*dff['ocorrencias'] / dff['populacao'],2)
        dff['dt_sintoma'] = [str(dff['dt_sintoma'][i]) for i in range(len(dff))]
        return(dff.to_dict('records'))


@app.callback(Output('filter1','children'), [Input('datatable-begin',"data")])
def filter_table1(value):
    if value is None:
        dff = dados
    else:
        dff = pd.DataFrame(value).sort_values(by=['dt_sintoma','UF','municipio'])
        return(html.Div())



if __name__ == '__main__':
    app.run_server(debug=True)