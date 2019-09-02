# standard library
from datetime import datetime as dt

# pydata stack
import pandas as pd

# dash libs
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash_table
from dash.dependencies import Input, Output

# internal libs
import IBGE as ib

# set params
pd.set_option('display.float_format', lambda x: '%.2f' % x)


###########################
# Data Manipulation / Model
###########################

# Read the main data file into a dataframe
dados = pd.read_csv('Dados/dengue_por_habitante.csv.gz')
dados = dados[dados['dt_sintoma'] >= '2006-01-01']

regioes = dados['regiao'].unique()
regioes.sort()

anos = dados['dt_sintoma'].unique()
anos.sort()

inicio = dados['dt_sintoma'].min()
fim = dados['dt_sintoma'].max()

# Read stationarity results
st = pd.read_csv('Dados/stationarity.csv')
st.columns = ['City', 'Test statistic', 'p-Value', 'Stationary']
st.drop('Stationary', axis=1, inplace=True)
st = st.round(decimals=3)

# Read cross-correlations and prepared dataframe
corrs = pd.read_csv('Dados/all_corrs.csv')
corrs.columns = ['municipio1', 'municipio2', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
corrs = pd.melt(corrs, id_vars = ['municipio1', 'municipio2'], var_name='lag', value_name='correlation')
corrs = corrs.round(decimals=2)

# Read the state capitals data into a dataframe
capitais = pd.read_csv('Dados/capitais.csv')
municipios = capitais['municipio'].unique()
municipios.sort()

# Slice dataframe by year
def slice_year(df, year1=inicio, year2=fim):
    return df[(df['dt_sintoma'] >= year1) & 
              (df['dt_sintoma'] <= year2)]

# Slice dataframe by region
def slice_region(df, region=['Sul', 'Sudeste', 'Nordeste', 'Norte', 'Centro-Oeste']):
    return df[df['regiao'].isin(region)]

# Count reports by region
def pct_region(df):
    return 100*(df.groupby(by='regiao').sum().ocorrencias/df.ocorrencias.sum())


#########################
# Dashboard Layout / View
#########################

# generate a table from a dataframe
def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


# Dash setup
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# create layout
app.layout = html.Div([
    
    # Page Header
    dcc.Markdown('''
        ## Degue Fever Epidemics: Using Distance as a Proxy in Forecasting
        ###### Lucas Ribeiro & Valter Moreno
        ###### Escola de Matemática Aplicada (EMAp) / FGV
        '''),
    
    # Text
    dcc.Markdown('''
        ### Introduction
        This project is part of the InfoDengue project developed by researchers of Fundação Getúlio
        Vargas and Fundação Oswaldo Cruz.
        
        ### Data
        The data analyzed in this project was obtained in the InfoDengue database. They consist of 
        a list of individual cases of dengue fever reported by different instances of the Brazilian 
        health system in the country's state capitals.
        
        ### Exploratory Data Analysis
        '''),
    
    dcc.Markdown('''
        #### Weekly reports of dengue fever in Brazilian state capitals

        **Select the date range to be displayed:**
        '''),
    
    # Pick dates
    dcc.DatePickerRange(
        id='year-range',
        min_date_allowed=inicio,
        max_date_allowed=fim,
        initial_visible_month=fim,
        start_date=inicio,
        end_date=fim,
        display_format='DD/MM/YYYY'
    ),
    # Line and bar charts of total records
    html.Div([
        dcc.Graph(
            id='ts-global'),
        dcc.Graph(
            id='ts-global-bar')
    ], style={'columnCount': 2}),

    # Line chart by region
    html.Div([
        html.Div(
            className='row',
            children=[
                html.H6('Weekly Reports of Dengue Fever in Brazilian State Capitals'),
                dcc.Checklist(
                    id='get-regs',        
                    options=[{'label': regiao, 'value': regiao} 
                            for regiao in regioes],
                    value=regioes,
                    labelStyle={'display': 'inline-block'}
                )
            ]
        ),
        html.Div(
            className='row',
            children=
                html.Div(
                    className='twelve columns',
                    children=
                        dcc.Graph(
                            id='ts-regs')
                    
                )
        )
    ]),
    
    # Per capta line chart by region    
    html.Div([
        html.Div(
            className='row',
            children=[
                html.H6('Weekly Reports of Dengue Fever per Capta in Brazilian State Capitals'),
                dcc.Checklist(
                    id='get-regs-hab',        
                    options=[{'label': regiao, 'value': regiao} 
                            for regiao in regioes],
                    value=regioes,
                    labelStyle={'display': 'inline-block'}
                )
            ]
        ),
        html.Div(
            className='row',
            children=
                html.Div(
                    className='twelve columns',
                    children=
                        dcc.Graph(
                            id='ts-regs-hab')
                    
                )
        )
    ]),
    
    dcc.Markdown(children='''
        ### Stationarity Test
        We used the [Augmented Dickey–Fuller test]
        (https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test)
        to assess the stationarity of each state capital's time series. Time series 
        cross-correlations may be inflated by trend and seasonality. It is important,
        therefore to check the stationarity of the series before calculating cross-correlations. 
        Below, we present the test results for the time series of the Brazilian capitals.
        '''
    ),
    
    # Plot stationarity table
    html.Div(
        children=generate_table(st)
    ),
  
    dcc.Markdown('''
        .
        
                 
        The following chart presents the cross-correlations between a selected state capital (the base city) and 
        the state capitals of selected regions.
        
        .             
        '''),
    
    # Select cities and plot cross-correlations
    html.Div([
        html.Div(
            className='row',
            children=
                html.H6('Cross-correlations between Time Series')
        ),
                 
        # Select cities         
        html.Div(
            className='row',
            children=[
                html.Div(
                    className='two columns',
                    children='Select the base city:'),
                html.Div(
                    className='two columns',
                    children=
                        dcc.Dropdown(
                            id='get-city',                
                            options=[{'label': city, 'value': city} for city in municipios],
                            value='Aracaju')
                )
            ]
        ),
        
        html.Div(
            className='row',
            children=[
                html.Div(
                    className='two columns',
                    children='Select regions:'),
                html.Div(
                    className='six columns',
                    children=
                        dcc.Checklist(
                            id='get-regions',        
                            options=[{'label': regiao, 'value': regiao} 
                                    for regiao in regioes],
                            value=regioes,
                            labelStyle={'display': 'inline-block'}
                        )
                )
            ]   
        ),
    
        # Cross-correlation chart:
        html.Div(
            className='row',
            children=
                dcc.Graph(
                    id='cross-corrs')
        )
    ])
])


# interaction callbacks

@app.callback(
    [Output('ts-global', 'figure'),
     Output('ts-global-bar', 'figure')],    
    [Input('year-range', 'start_date'),
     Input('year-range', 'end_date')])
def update_line_bar(start_date, end_date):
    df_years = slice_year(dados, start_date, end_date)
    df = df_years.groupby('dt_sintoma')['ocorrencias'].sum()
    df = df.reset_index()
    pct_years = pct_region(df_years)
    
    line_figure = {
        'data': [
            go.Scatter(
                x = df['dt_sintoma'],
                y = df['ocorrencias'],
                mode = 'lines',
                line=dict(color='firebrick'),
                showlegend = False
            )],
        'layout': go.Layout(
                template='plotly_white',
                title= 'Total Number of Dengue Fever Reports in Brazilian State Capitals',            
                xaxis={'title': 'Date'},
                yaxis={'title': 'Number of Reports'},
                hovermode='closest'
                )
        }
    
    bar_figure = {
        'data': [
            {'x': regioes,
             'y': pct_years, 
             'type': 'bar',
             'hovertext': [str(round(i, 2))+'%' for i in pct_years]}
            ],
        'layout': {
            'title': 'Percentage of total number of reports by region'
            }
        }

    return line_figure, bar_figure

@app.callback(
    Output('ts-regs', 'figure'),
    [Input('get-regs', 'value')])
def update_regs(value):
    df = slice_region(dados, value)
    regs_figure = {
        'data': [
            go.Scatter(
                x = df[df['co_municipio'] == i]['dt_sintoma'],
                y = df[df['co_municipio'] == i]['ocorrencias'],
                text = df[df['co_municipio'] == i]['municipio'],
                mode = 'lines',
                name = df[df['co_municipio'] == i]['municipio'].iloc[0] 
            ) for i in df['co_municipio'].unique()
        ],
        'layout': go.Layout(
            template='plotly_white',            
            xaxis={'title': 'Date Range'},
            yaxis={'title': 'Weekly Reports'},
            height=600,
            hovermode='closest',
            xaxis_rangeslider_visible=True
        )
    }
    return regs_figure

@app.callback(
    Output('ts-regs-hab', 'figure'),
    [Input('get-regs-hab', 'value')])
def update_regs_hab(value):
    df = slice_region(dados, value)    
    regs_hab_figure = {
        'data': [
            go.Scatter(
                x = df[df['co_municipio'] == i]['dt_sintoma'],
                y = df[df['co_municipio'] == i]['por_habitante'],
                text = df[df['co_municipio'] == i]['municipio'],
                mode = 'lines',
                name = df[df['co_municipio'] == i]['municipio'].iloc[0]
            ) for i in df['co_municipio'].unique()
        ],
        'layout': go.Layout(
            template='plotly_white',
            xaxis={'title': 'Date Range'},
            yaxis={'title': 'Weekly Reports per Capta'},
            height=600,
            hovermode='closest',
            xaxis_rangeslider_visible=True
        )
    }    
    return regs_hab_figure

@app.callback(
    Output('cross-corrs', 'figure'),
    [Input('get-city', 'value'),
     Input('get-regions', 'value')])
def update_crosscorr(base, regions):
    if len(regions) <= 1:
        regions = list(regions)
    
    cities = dados[dados['regiao'].isin(regions)]['municipio'].unique()
        
    df = corrs[(corrs['municipio1'] == base) &
               (corrs['municipio2'].isin(cities))]
    
    corrs_figure = {
        'data': [
            go.Scatter(
                x = df[df['municipio2'] == i]['lag'],
                y = df[df['municipio2'] == i]['correlation'],
                mode = 'lines',
                text = df[df['municipio2'] == i]['municipio2'],
                name = df[df['municipio2'] == i]['municipio2'].iloc[0]
            ) for i in df['municipio2'].unique()
        ],
        'layout': go.Layout(
            template = 'plotly_white',            
            xaxis={'title': 'Lag (in weeks)'},
            yaxis={'title': 'Correlation'},
            hovermode='closest',
            height=600
        )
    }
    
    return corrs_figure
    
                
if __name__ == '__main__':
    app.run_server(debug=True)