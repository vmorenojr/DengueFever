# standard library
from datetime import datetime as dt

# pydata stack
import pandas as pd

# dash libs
import plotly.express as px
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
    html.H3('Introduction'),        
        
    html.Div(
        className='row',
        children=[
            dcc.Markdown(
                className='eight columns',
                children='''
                This project is part of the InfoDengue project developed by researchers of Fundação Getúlio
                Vargas and Fundação Oswaldo Cruz.
                '''
            ),
            html.Iframe(
                className='four columns',
                #width=400,
                height=280,
                src='https://www.youtube.com/embed/Wzb4cIFc36g?list=WL',
                style={'frameborder':'0',
                    'allow':'accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture'}
            )
        ]
    ),
    
    html.Br(),
    html.Br(),
    
    html.Div(
        className='row',
        children=[
            dcc.Markdown(
                className='eight columns',
                children='''
                Dengue in Brazil.
                '''
            ),
            html.Iframe(
                className='four columns',
                #width=500,
                height=280,
                src='https://www.youtube.com/embed/7gU0XBhXIS4?list=WL',
                style={'frameborder':'0',
                    'allow':'accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture'}
            )
        ]
    ),
    
    html.Br(),
    html.Br(),
    
    html.H3('Data'),
    
    dcc.Markdown('''                
        The data analyzed in this project was obtained in the InfoDengue database. They consist of 
        a list of individual cases of dengue fever reported by different instances of the Brazilian 
        health system in the country's state capitals.
        '''
    ),
    
    html.Br(),
    html.Br(),
    
    html.H3('Exploratory Data Analysis'),
    
    html.Br(),
    html.Br(),
    
    html.H4('Reports of Dengue Fever in Brazilian State Capitals'),
    
    dcc.Markdown('''
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
    # Line and bar charts of total reports
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
                    className='ten columns',
                    children=
                        dcc.Graph(
                            id='ts-regs')
                    
                )
        )
    ]),
    
    html.Br(),
    html.Br(),
    
    # Per capta line chart by region    
    html.Div([
        html.Div(
            className='row',
            children=[
                html.H6('Weekly Reports per 100k Population in Brazilian State Capitals'),
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
                    className='ten columns',
                    children=
                        dcc.Graph(
                            id='ts-regs-hab')
                    
                )
        )
    ]),
    
    html.Br(),
    html.Br(),
    
    html.H4('Cross-correlations Analysis'),
    
    html.H6('Stationarity Tests'),
    
    dcc.Markdown(children='''
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
  
    html.Br(),
    html.Br(),
    
    dcc.Markdown('''
        The following chart presents the cross-correlations between a selected state capital (the base city) and 
        the state capitals of selected regions.
        '''),
    
    html.Br(),
    html.Br(),
    
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
                            value=['Nordeste'],
                            labelStyle={'display': 'inline-block'}
                        )
                )
            ]   
        ),
    
        # Cross-correlation chart:
        html.Div(
            className='row',
            children=[
                html.Div(
                    className='eight columns',
                    children=
                        dcc.Graph(
                            id='cross-corrs')
                ),
                html.Div(
                    className='four columns',
                    children = [
                        html.Br(),
                        dcc.Markdown('''
                            The chart suggests that, at least for some cities, distance
                            does play a role in the spread of the epidemic. Most cross-correlations
                            drop with the lag for nearby cities. In some cases of farther located
                            cities, the cross-correlations reach their maximum value later on, 
                            for lags greater than four weeks.
                            ''')
                    ]
                )
            ]
        )
    ]),
    
    html.Br(),
    html.Br(),
    
    # Map with animation
    html.H4('The Dynamics of Dengue Fever Epidemics in Brazilian State Capitals'),
    
    html.Div(
        className='row',
        children=[
            html.Div(
                className='six columns',
                children=[
                    html.Br(),
                    html.Br(),
                    dcc.Markdown('''
                        The map to the right displays the montly reports of dengue fever per 100,000
                        population in Brazilian state capitals since August, 2012. 
                        The dynamics of the spread of the desease over time is shown by clicking
                        the play button or by moving the slide laterally.
                        
                        The epidemic seems to start in the central states capitals of the country, and 
                        move to the Southeast and Northeast. Apparently, farther cities from
                        the central states tend to develop the epidemic later than nearby cities.
                        ''')
                ]
            ),
            html.Div(
                className='six columns',
                children=
                    dcc.Graph(id='mapa', 
                        style={'width': '75%', 'display': 'inline-block'},
                        figure=px.scatter_geo(dados_mes, 
                                    scope='south america',
                                    lat='latitude', lon='longitude',
                                    hover_name='municipio', 
                                    size='por_habitante',
                                    animation_frame='dt_sintoma',
                                    projection='mercator')
                    )
            )
        ]
    )
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
                title= 'Total Number of Reports',            
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
            'title': 'Percentage of Reports by Region'
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
            margin={'l':5, 'r':5, 'b':5, 't':20, 'pad':2}#,
            #xaxis_rangeslider_visible=True
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
            margin={'l':5, 'r':5, 'b':5, 't':20, 'pad':2},
            hovermode='closest'#,
            #xaxis_rangeslider_visible=True
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
            height=600,
            legend=dict(x=-.2, y=1),
            margin={'l':5, 'r':5, 'b':5, 't':20, 'pad':2}
        )
    }
    
    return corrs_figure
    
                
if __name__ == '__main__':
    app.run_server(debug=True)