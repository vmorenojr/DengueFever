# standard library
from datetime import datetime as dt
import base64

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

# Total number of cases in each city
total_ocorrencias = dados.loc[dados.dt_sintoma >='2016-05-05', ['municipio', 'ocorrencias']]\
                         .groupby('municipio')\
                         .sum()\
                         .round(decimals=2)\
                         .reset_index()\
                         .sort_values(by='ocorrencias', ascending=False)
total_ocorrencias.columns = ['City', 'Total number of cases']

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

distances = pd.read_csv('Dados/distancias.csv')

# Read XGBoost results
bh_lags = pd.read_csv('XGB_fit/baseline-lags.csv')
bh_caps = pd.read_csv('XGB_fit/allcaps.csv')
bh_caps_no_time = pd.read_csv('XGB_fit/allcapsnotime.csv')
bh_imp = pd.read_csv('XGB_fit/importances.csv')
bh_imp = bh_imp.merge(distances[['municipio','Belo Horizonte']], how='left',
                      left_on='City', right_on='municipio')\
               .drop('municipio', axis=1)\
               .rename(columns={'Belo Horizonte': 'Distance'})

# -------------------
# Auxiliary functions
# -------------------

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

# Convert PNG image
def convert(image_file):
    return base64.b64encode(open(('Plots/' + image_file), 'rb').read()).decode('ascii')


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
    html.H2('Predicting Dengue Fever Epidemics in Brazilian State Capitals'),
    html.H3('Lucas Ribeiro & Valter Moreno', style={'color': '#566573'}),
    html.H5('Escola de Matemática Aplicada (EMAp) / FGV', style={'color': '#566573'}),
    html.Br(),
    html.Br(),
    
    # Text
    html.H3('Introduction'),
        
    html.Div(
        className='row',
        children=[
            dcc.Markdown(
                className='eight columns',
                children='''
                This project is part of the [InfoDengue project](https://doi.org/10.1101/046193) developed by 
                [Prof. Flavio Codeço](https://emap.fgv.br/corpo-docente/flavio-codeco-coelho) and colleagues 
                from Fundação Getúlio Vargas ([FGV](http://emap.fgv.br/)) and Fundação Oswaldo Cruz ([Fiocruz](https://portal.fiocruz.br)). 
                In addition to its various research-oriented deliveries, the project has implemented an online dashboard that is 
                weekly updated with Dengue, Chikungunya and Zika fever data from 790 cities in Brazil. In this way, it enables local health
                agencies to make better decisions as far as planning and the allocation of resources is concerned, and subsenquently evaluate
                the outcomes of its efforts.
                
                Our project focuses on the dynamics of Dengue fever epidemics in the 27 Brazilian state capitals. Its main objective is 
                to assess the possibility of using the distances among those cities and the historical data on Dengue fever reports to predict
                future epidemics. Different machine learning (ML) techniques were used to make the predictions. Their accuracy results were compared
                and assessed, and the practical implications of our findings, discussed.
                
                According to the World Health Organization ([WHO](https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue)),
                Dengue fever, whose primary vector is the Aedes aegypti mosquito, has spread globally in the recent decades. 
                As the majority of cases are asymptomatic, official reports tend to underestimate the number of people infected 
                with the Dengue viruses. However, in 2012, [Brady et al. (2012)] (https://doi.org/10.1371/journal.pntd.0001760) had
                already estimated that 3.9 billion people in 128 countries were at risk of infection.
                
                The WHO stresses the the alarming impact of Dengue fever on both human health and the global and national economies. Such impact
                is intensified by explosive outbreaks, which have been increasingly more frequent. Among travellers returning from low- and
                middle-income countries, dengue has become the second most diagnosed cause of fever after malaria. The forecasted climatic changes
                can drastically worsen this situation, as it is discussed in the video to the right.
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
    
    html.H5('Dengue Fever in Brazil'),
    
    html.Div(
        className='row',
        children=[
            dcc.Markdown(
                className='eight columns',
                children='''
                The World Health Organization (WHO) reports that, in 2016, in one of the largest worldwide outbreaks of dengue,
                the Americas region registered more than 2.38 million cases, "where Brazil alone contributed slightly less
                than 1.5 million cases, approximately 3 times higher than in 2014".
                
                In a [recent report](https://portalarquivos2.saude.gov.br/images/pdf/2019/agosto/13/Informe-Arboviroses-SE-30.pdf),
                the Brazilian Health Ministry states that there were 1,393,062 probable cases and 914,310 confirmed cases of Dengue
                in the country between January and July, 2019. Even more worrisome is the reported **increase of 610.6%** of 
                probable cases when compared to the same period in the previous year. The video to the right shows 
                a news report on this issue.
                
                The spread of Dengue fever in Brazil represents and additional burden on the already depleted financial 
                resources of the country. According to the 
                [Brazilian Health Ministry](http://www.saude.gov.br/noticias/agencia-saude/45257-ministerio-da-saude-alerta-para-aumento-de-149-dos-casos-de-dengue-no-pais),
                the prevention efforts have been consuming a fastly growing amount of money: from R$ 924.1 million in 2010 to R$ 1.73 billion in 2018.
                This funding is transferred monthly to state and local governments to be spent with the monitoring contagious diseases, including Dengue, Zika
                and Chikungunya. Therefore, the aforementioned numbers do not include the amout spent with the treatment of patients infected with the
                Dengue viruses.
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
        The main source of data used in this project was the InfoDengue database. We were able to 
        obtain a CSV file with a list of individual cases of Dengue fever reported in the country's 
        state capitals from January, 2006 to May, 2019. It contained 3,684,688 records in total.
        
        The following fields were included in file:
          - __*dt_notificacao*__: the date when the case was officially reported by a health agency
          - __*co_cid*__: the WHO international code for the Dengue fever
          - __*co_municipio_notificacao*__: the code of the city in which the case was officially reported
          - __*co_municipio_residencia*__: the code of the city where the patient lived
          - __*dt_diagnostico_sintoma*__: the date reported by the patient when the symptoms started
          - __*dt_digitacao*__: the date when the case was inputed in the system
         
        The main dataset had no missing values and was mostly clean. Still, we had to 
        address two issues: the definition of the date and location for each Dengue fever case. 
        
        Given the potential delay between the time when a person develops the symptoms of Dengue and
        the time he/she is able to have an appointment in a hospital or clinic, we decided to use
        the self-reported date when the symptoms started as the reference date for the infection, and
        discard the other dates.
        
        At first, we were going to use the code of the city where the case was officially reported to
        identify the state capital where the infection occurred. In Brazil, people often have to travel
        to larger cities to have access to better health services. Thus, we expected to find a larger
        variety of codes in the co_municipio_residencia than in the co_municipio_notificacao field.
        Surprisingly, upon close examination, we noticed that the co_municipio_notificacao field 
        contained codes for most of the cities in Brazil. On the other hand, the co_municipio_residencia
        included codes only for the Brazilian state capitals. After consulting with the head of the 
        InfoDengue project, we decided to use the co_municipio_residencia field to identify the 
        state capital in which a case occurred.
        
        Given the large number of reported cases, it was necessary to aggreate the number of cases by
        a time period (day, week, month). We decided to aggregate the data by weeks to be able to
        have an adequate balance between the number of observations and the usage of computational
        resources in our analyses. The aggregated dataset had 115,203 observations.
        
        We used data provided by the Brazilian Institute of Geography and Statistics (IBGE) to 
        associate the code of a city to its name and state, as well as to obtain its latitude 
        and longitude. In addition, we scraped the acronyms of the states and their regions
        (Norte, Nordeste, Centro-oeste, Sudeste, Sul) from the [Estados e Capitais do Brasil]
        (https://www.estadosecapitaisdobrasil.com/) website. 
        
        As we wanted to use the number of reported cases of Dengue per 100,000 population in 
        our analyses, it was necessary to obtain data on the population of each state capital
        from 2006 to 2019. We were unable to find the complete information in websites or official
        reports. [Wikipedia](https://pt.wikipedia.org/wiki/Lista_de_capitais_do_Brasil_por_população)
        proved to be the best reliable source, but had data only for 2000, 2010 and 2019. 
        We scraped this information and used a smooth spline approximation to fill in the values
        for each week in the main dataset.
        '''
    ),
    
    html.Br(),
    
    html.H3('Exploratory Data Analysis'),
    dcc.Markdown('''                
        In the next sections, we present several interactive charts in which one can
        explore the evolution of Dengue fever in Brazilian state capitals over time.
        It is possible to select the region of the country for which data will be 
        displayed, as well as to clik on individual cities in the legend to hide or
        show its data. Other features include zooming in and out and sliding the time
        axis.
        
        Overall, it is possible to see that Dengue fever epidemics follow the cycles of 
        rainy and dry seasons (equivalent to summer/autumn and winter/spring) in the coutry. 
        Furthermore, we notice that the largest proportion of overall cases is in the 
        Sourtheast region, which concentrates most of the population of Brazil. In the
        Center-West region, Goiania seems to be the capital most affected, while in the
        Northeast, Fortaleza has registered the highest number of cases. When the 
        number of cases per 100,000 population is considered, a similar pattern emerges.
        
        It is interesting to see that the total number of cases has fallen in 2017 and
        2018, only to sharply increase again in 2019. This is in line with the abovementioned
        video, which discusses the lack of investments in prevention and its recent consequences. 
        
        '''
    ),
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
                    value=['Centro-Oeste'],
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
                    value=['Centro-Oeste'],
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
    
    html.Br(),
    html.Br(),
    
    html.H4('Cross-correlations Analysis'),
    
    dcc.Markdown(children='''
        Cross-correlations between time-series indicate how much they are related to each other.
        If the occurrence of Dengue in one region is related to the number of cases in another, 
        their time series should be correlated. In addition, it would be reasonable to expect
        cross-correlations between lagged time-series to increase with the lag for cities that
        are farther from each other, and to decrease for cities that are closer to each other. 
        Thus, the examination of lagged cross-correlations help us understand the role of 
        distance in the dynamics of Dengue fever epidemics.
        '''
    ),
        
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
    
    # Table: Stationarity
    html.Div(
        className='row',
        children=[
            html.Div(
                className='four columns',            
                children=generate_table(st.iloc[:14], max_rows=16)
            ),
            html.Div(
                className='four columns',            
                children=generate_table(st.iloc[14:], max_rows=16)
            )
        ],
        style={'textAlign': 'center'}
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
                            value=['Centro-Oeste'],
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
                            
                            Similar patterns are shown for cities in each region of the country. It is
                            possible that such patterns are related with the flow of people among
                            state capitals. Accurate data on travel patterns are not available in 
                            Brazil, unfortunately.
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
                        the play button or by moving the slide laterally. It is also possible to
                        zoom in and out, and to move the map within its window.
                        
                        The epidemic seems to start in the central states capitals of the country, and 
                        move to the Southeast and Northeast. Apparently, farther cities from
                        the central states tend to develop the epidemic later than nearby cities. This
                        pattern seems to be in accordance with our examination of the time series 
                        cross-correlations. Thus, an assessment of the role of distance in Dengue
                        fever epidemics through machine learning models seems to be warranted.
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
    ),
    
    html.H3('Predicting Dengue Fever Epidemics'),
    
    # Data preparation
    html.H4('Data Preparation'),
    
    html.Div(
        className='row',    
        children=[
            dcc.Markdown('''
                The datasets used for training, testing and validating our models was created from the historical 
                dengue fever reports database obtained from InfoDengue. We created a dataset for each city, with
                the following features:
                - __*data_alvo*__: the week for which we want to predict the number of reports of Dengue fever
                - __*year*__: year of *data_alvo*
                - __*quarter*__: quarter of *data_alvo*
                - __*month*__: month of *data_alvo*
                - __*dayofmonth*__: day of the month for *data_alvo*
                - __*dayofyear*__: day of the year for *data_alvo*
                - __*dayofweek*__: day of the week for *data_alvo*
                - __*outcome*__: either number of Dengue fever reports or number of Dengue fever reports per 100,000 population to be predicted
                - __*municipio_i*__: number of reports in the state capital whose name is municipio (including the target
                city) with a lag of i weeks before *data_alvo*
                
                We used all features in the dataset, except for *data_alvo* as predictors of *outcome*.
                
                We wanted to consider the proximity of the Brazilian state capitals in our analysis. 
                We chose initially to compute the distance between the cities using possible
                land paths instead of the great circle distance (airflight distance), as the vast majority
                of people in Brazil travel by bus or car. 
                
                We used the Google Maps API to obtain the distances. First, we got an API key from Google API services
                to use it with the googlemaps library in python. Then, for every state capital, we requested the distances
                to other capitals from Google Maps. The data was stored in a distance matrix.
                
                It is important to mention that Google Maps could not find any valid route between the capitals of 
                Amapá and Acre and any other Brazilian state capitals. Because of this, we reverted to use the 
                great circle distances, which we computed using the Haversine formula. The formula takes has as input the
                coordinates of the cities, and generates the distance in kilometers of the great
                circle between the cities.                
                
                Due to computer resources constraints and the availability of data, the following time periods
                were used to train, test and validate the models:
                - __*data_alvo*__: May 5, 2019 to May 5, 2010
                - __*municipio_i*__: we considered all weeks in intervals of 8, 12, 26 and 52 weeks (i.e., 
                two months, three months, six months and one year) starting from 4, 12, 26 and 52 weeks (i.e., 
                one month, three months, six months and one year) before *data_alvo*
                years
                
                The choice of lags was based on potential practical applications of our analysis. We believed
                the predictions generated by our model could be used for planning purposes as far as the 
                allocation of resources to prevent Dengue fever is concerned. The predictions would probably be
                more useful the longer time in advance they were made and the more accurate they were. One month
                seems to be a reasonable threshold for actionable decision-making in the case of the prediction
                or mitigation of epidemics.
                '''
            )
        ]
    ),
    
    # Modeling
    html.H4('Modeling'),
    
    html.Div(
        className='row',    
        children=
            dcc.Markdown('''
                We split the datasets in a training set containing historical data until May 05, 2018, and a 
                testing set, with data from May 05, 2018 to May 05, 2019. To improve accuracy, individual models
                were generated for each city. In a country as big as Brazil, distant cities are prone to be
                subjected to the influence of different factors that were not included in our datasets.
                                
                Due to computer resources limitations, we chose to conduct our analysis only for the city
                of Belo Horizonte, the Minas Gerais state capital. As can be seen in the exploratory data analyisis,
                Belo Horizonte is one of the cities with the largest number Dengue fever cases in the last years,
                and one of the most important state capitals in the coutry. The total number of reported Dengue fever 
                cases in each Brazilian state capital since May 05, 2016 is reported below.                               
                ''')
    ),
    
    # Table: total cases in each city
    html.Div(
        className='row',
        children=[
            html.Div(
                className='four columns',            
                children=generate_table(total_ocorrencias.iloc[:14],
                                        max_rows=16)
            ),
            html.Div(
                className='four columns',            
                children=generate_table(total_ocorrencias.iloc[14:],
                                        max_rows=16)
            )
        ],
        style={'textAlign': 'center'}
    ),
    html.Br(),
            
    dcc.Markdown('''
        Two machine learning techniques were used to predict Degue fever epidemics: **XGBoost** and
        **Neural Networks**.
        '''),
    html.Br(),
    
    # XGBoost
    html.H5('XGBoost'),

    dcc.Markdown('''
        XGBoost is an optimized gradient-boosting machine learning (ML) library 
        that has APIs in several languages, including Python. Its popularity is due not 
        only to its superior speed and support for parallelization, but also its
        excellent performance. XGBoost has consistently outpeformed most of the
        single-algorithm methods available to this point in many ML tasks.
        
        Our searches in the literature and the Web suggested that the tree-based 
        XGBoost usually shows a better performance than the linear algorithm. 
        We used the first option in our analysis and adopted the RMSE as our 
        performance measure to evaluate our models.
        '''),
    html.Br(),
    
    html.H6('Exploratory analyis'),
    
    # Baseline
    dcc.Markdown('''    
        To have an overal understanding of the accuracy of our model, we initially compared its 
        outcomes with those generated with a model with only time-related information (year, quarter,
        month, etc.), and a model with time-related information and lagged data only for Belo Horizonte.
        We run the analyses with the default settings of XGBoost.
        '''),
    html.Br(),
    
    dcc.Markdown('''    
        Below, we show the original time series, whith the training data in blue and the test date in 
        orange. In addition, we show the results obtained with the baseline model, which included only
        the date-related features as predictors. It is clear that the predictions were quite off the 
        actual values. The fit results we obtained were:
        - __*RMSE*__: 1839.45
        - __*MAE (Mean Absolute Error)*__: 773.34
        - __*MAPE (Mean Absolute Percentage Error)*__: 146.88
        '''),
    html.Br(),
    
    dcc.Markdown('''    
        Of the time-related predictors, year, day of the year and week of the year were by far the 
        most important. This may be due to the rainy and dry seasons cycles not conforming exactly to 
        months, thereby making the number of the month in a year and the day of the month less 
        relevant to prediction.  
        '''),
    
    # Charts: Baseline
    html.Div(
        className='row',
        children=[
            html.Div(
                className='five columns',
                children=[
                    html.Br(),                
                    html.Img(src='data:image/png;base64,{}'.format(convert('ts-baseline.png')),
                             width=500, height=250),
                    html.Img(src='data:image/png;base64,{}'.format(convert('pred-base.png')),
                             width=500, height=250)
                ]
            ),
            html.Div(
                className='seven columns',
                children=
                    html.Img(src='data:image/png;base64,{}'.format(convert('imp-base.png')),
                             width=780, height=520)
            )
        ]
    ),
    html.Br(),      
    
    dcc.Markdown('''    
        Next, we examined models with lagged data on the occurrence of Dengue in Belo Horizonte only,
        and with lagged data on the occurrence of Dengue in all state capitals, including Belo
        Horizonte. Different lags and ranges of lagged ocurrences were tested. The following tables
        summarize the fit results.
        '''),
    html.Br(),
    
    # Table: Fit results for Belo Horizonte for lagged outcomes
    html.Div(
        className='row',
        children=[
            html.Div(
                className='six columns',
                children=[
                    dcc.Markdown('''**Lagged outcomes for Belo Horizonte only**'''),
                    generate_table(bh_lags.drop('All cities', axis=1)\
                                               .round(decimals=2), max_rows=20)
                ]
            ),  
            html.Div(
                className='six columns',
                children=[
                    dcc.Markdown('''**Lagged outcomes for all state capitals**'''),
                    generate_table(bh_caps.drop('All cities', axis=1)\
                                               .round(decimals=2), max_rows=20)
                ]
            )
        ]
    ),
    html.Br(),
    html.Br(),
    
    dcc.Markdown('''    
        When compared with the baseline results, only those for a lag of four weeks showed 
        substantial improvements. Oddly, when we compare the models with lagged outocmes for
        Belo Horizonte only and for all state capitasl, the latter's results were consistently
        better than the former's only for the 4-week lag. It is important to remember that
        the second model includes the data on Belo Horizonte itself, as well as on the other
        state capitals. Thus, it predictors contain all the information contained in the
        predictors of the first model and more. This issue may be a result of the way the
        XGBoost algorithm works and/or the default settings used in our exploratory 
        analysis.
        
        It is also interesting to note that the best results for RMSE do not correspond
        to the best results for MAPE. This applies to the results obtained for all lags.
        The cause behind this apparent discrepancy is that RMSE is a measure of absolute
        error, while MAPE is a measure of relative error. Thus, the value for MAPE may 
        be high even when the absolute error (and thus RMSE) is low. We should note that
        the number of cases of Dengue fever is very low for many of the weeks in our test
        dataset. In such weeks, the squareroot of the squared error tend to be low, while
        the absolute percent error may be quite high.
    '''),
    dcc.Markdown('''    
        The following charts compare the actual Dengue fever occurrences with
        the predicted values for the models with data on Belo Horizonte only,
        and on all state capitals. We selected the charts corresponding to the best RMSE 
        result for each lag and each model.
        '''),
    html.Br(),
    
    # Charts: Belo Horizonte with lags
    html.Div(
        className='row',
        children=[
            html.Div(
                className='six columns',
                children=[
                    dcc.Markdown('''__*Lagged outcomes for Belo Horizonte only*__'''),
                    html.Br(),
                
                    dcc.Markdown('''**Lag: 4 weeks, Range: 1 year**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('pred-base-l4r52.png')),
                             width=600, height=300),
                    
                    dcc.Markdown('''**Lag: 12 weeks, Range: 1 year**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('pred-base-l12r52.png')),
                             width=600, height=300),
                    
                    dcc.Markdown('''**Lag: 6 months, Range: 12 weeks**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('pred-base-l26r12.png')),
                             width=600, height=300),
                    
                    dcc.Markdown('''**Lag: 1 year, Range: 12 weeks**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('pred-base-l52r12.png')),
                             width=600, height=300)
                ]
            ),
            html.Div(
                className='six columns',
                children=[
                    dcc.Markdown('''***Lagged outcomes for all state capitals***'''),
                    html.Br(),
                    
                    dcc.Markdown('''**Lag: 4 weeks, Range: 6 months**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('pred-all-l4r26.png')),
                             width=600, height=300),
                    
                    dcc.Markdown('''**Lag: 12 weeks, Range: 6 months**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('pred-all-l12r26.png')),
                             width=600, height=300),
                    
                    dcc.Markdown('''**Lag: 6 months, Range: 8 weeks**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('pred-all-l26r8.png')),
                             width=600, height=300),
                    
                    dcc.Markdown('''**Lag: 1 year, Range: 6 months**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('pred-all-l52r26.png')),
                             width=600, height=300)
                ]
            )
        ]
    ),
    html.Br(),   
    html.Br(),
    
    dcc.Markdown('''    
        As expected, the charts for the lower values of lag show predictions closer to the
        actual values. As the RMSE results suggested, the best fit were obtained with a
        lag of four weeks. We will focus our subsequent analyses on models based on this
        value of lag.        
        '''),
    dcc.Markdown('''    
        First, however, we will use the importance plot generated by XGBoost to check the features
        that were more relevant in the precdiction process. The following charts show the 30 most
        important features for the two models.       
        '''),
    html.Br(),
    
    # Charts: importance plots
    html.Div(
        className='row',
        children=[
            html.Div(
                className='six columns',
                children=[
                    dcc.Markdown('''**Lag: 4 weeks, Range: 1 year, Belo Horizonte only**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('imp-base-l4r52.png')),
                             width=540, height=360)
                ]
            ),
            html.Div(
                className='six columns',
                children=[
                    dcc.Markdown('''**Lag: 4 weeks, Range: 6 months, all state capitals**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('imp-all-l4r26.png')),
                             width=540, height=360)
                ]
            )
        ]
    ),
    html.Br(), 
    
    dcc.Markdown('''    
        The first time-related feature in the importance plot for the model with data on
        Belo Horizonte only is the day of the yaer. In addition to it, week of the year
        and year were among the 10 most imporant features for the prediction process.
        
        In contrast, the second chart shows no time-related features at all. Instead,
        all listed features correspond to lagged outcomes for various cities. Belo
        Horizonte itself is shown at the 21st position. In addition, eight or the
        ten most important features are lagged outcomes for *Campo Grande* and *Aracaju*.
        This finding may be related to the flow of travellers between Minas Gerais
        state capital and those two cities.
        
        In any case, the importance plot for the second model suggests that 
        the lagged time series for state capitals actually contain most of the 
        information that is present in the time-related features  and that is useful
        for predicting the occurrence of Dengue fever in Belo Horizonte. 
        In addition, it is important to note that those time series implicitly 
        contain information on the distance between the target state apital and
        other state capitals. So the influece of a lagged outcome feature on the 
        predicted outcome actually combines the influences of time, distance,
        and the intensity of Dengue fever.
        
        Before we move to the next phase of our investigation, we will compare the
        results generated with the model with lagged outcomes for all state capitals
        as well as time-related features, and the same model with no time-related 
        features.
        '''),
    html.Br(),
    
    # Table: Fit results with and without time-related features
    html.Div(
        className='row',
        children=[
            html.Div(
                className='six columns',
                children=[
                    dcc.Markdown('''**Lagged outcomes with time-related features**'''),
                    generate_table(bh_lags[bh_lags['Lag (weeks)']==4]\
                                        .drop('All cities', axis=1)\
                                        .round(decimals=2), max_rows=20)
                ]
            ),  
            html.Div(
                className='six columns',
                children=[
                    dcc.Markdown('''**Lagged outcomes with no time-related features**'''),
                    generate_table(bh_caps_no_time.drop('All cities', axis=1)\
                                                  .round(decimals=2), max_rows=20)
                ]
            )
        ]
    ),
    html.Br(),
    html.Br(),
    
    # Charts: Fit results with and without time-related features
    html.Div(
        className='row',
        children=[
            html.Div(
                className='six columns',
                children=[
                    dcc.Markdown('''**Lag: 4 weeks, Range: 6 months, with time-related features**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('pred-all-l4r26.png')),
                             width=540, height=360)
                ]
            ),
            html.Div(
                className='six columns',
                children=[
                    dcc.Markdown('''**Lag: 4 weeks, Range: 6 months, no time-related features**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('pred-all-no-time-l4r26.png')),
                             width=540, height=360)
                ]
            )
        ]
    ),
    html.Br(),
    
    dcc.Markdown('''    
        Remarkably, the second table shows that the elimination of the 
        time-related predictors actually improved the RMSE for all the
        ranges. Accordingly, the second chart, which was generated with
        the combination of lag and range with the lowest RMSE, 
        does show an improvement in fit, although it may seem minor
        at first.
        
        Given these results, we will focus our efforts of a model with 
        a lag of four weeks, a range of six months, and no time-related 
        features.               
        '''),
    html.Br(),
    
    html.H6('Tuning the hyperparameters'),
      
    dcc.Markdown('''
        The following steps were used to train and define the hyperparameters (
        [XGBoost, 2019](https://xgboost.readthedocs.io/en/latest/parameter.html);
        [Jain, 2016](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/);
        [Cambridge Spark, 2017](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f)):
        '''),
    
    html.Div(
        className='row',
        children=[
            html.Div(
                className='one column'
            ),
            html.Div(
                className='eleven columns',
                children=
                    dcc.Markdown('''
                        First, we chose a learning rate of 0.3 and fit models for different number of trees.
                        The remaining hyperparameters were set to their defaults.
                                        
                        Second, we tuned two of the most important tree-specific parameters, *max_depth* and
                        *min_child_weight* for the selected learning rate and the number of trees found in the
                        previous step.
                                        
                        Third, we tried to reduce model complexity and enhance performance by tuning the
                        parameters *gamma*, *subsample*, and *colsample_bytree*.
                                            
                        Finally, we lowered the learning rate to define the optimal set of parameters.
                    ''')
            )
        ]
    ),
    
    dcc.Markdown('''     
        As pointed out by [Cochrane, 2018]
        (https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9),
        cros-validation methods, such as k-folding, should not be employed to validate
        time series models. Instead, she suggests the usage of a nested cross-validation
        procedure, which "provides an almost unbiased estimate of the true error"
        (Varma & Simon, 2006 apud Cochrane, 2018). In addition, to reduce the chances
        of getting a biased estimate of prediction error, Chochane argues that the
        validation and test procedure should employ the 
        rolling-origin-recalibration method (Bergmeir and Benítez, 2012).
        
        In the following analyses, we followed Cochrane's suggestion, adopting a
        period of 12 weeks as the unit for the rolling-origin-recalibration 
        method. Thus, as our dataset comprised 469 weeks, we had 39 time units for 
        testing purposes.
        
        Our findings are reported in more detail below.
    '''),
    
    # Number of trees
    dcc.Markdown('''
        ** Number of trees:**
        
        We followed the guidelines obtained in the above-mentioned documents
        and defined the following initial values for our parameters:
        - __*learning_rate*__: .3
        - __*min_child_weight*__: 1          
        - __*max_depth*__: 6
        - __*gamma*__: 0
        - __*subsample*__: 1
        - __*colsample_bytree*__: .75
         
        We set maximum number of boosting rounds to values varying from 500 to
        10000, with an early stopping at 50 rounds. The changes in the hyperparameter
        had a very low effect on the RMSE, which remained pratically constant at
        125.8. We set it to the minimum value we tested, that is, 500. 
    '''),
    html.Br(),
    
    # max_depth and min_child_weight 
    dcc.Markdown('''
        **Maximum depth of a tree and minimum weight in a child**
        
        The XGBoost documentation informst that increasing the value of *'max_depth'*
        will make the model more complex and more likely to overfit. It also warns 
        that XGBoost aggressively consumes memory when training a deep tree.
        
        The minimum sum of instance weight needed in a child is defined by the 
        *'max_depth'* hyperparameter. According to the XGBoost documentation, the
        partitioning process will stop when the tree partition step results in a 
        leaf node with the sum of instance weight less than min_child_weight. 
        Thus, the larger its value is, the more conservative the algorithm will be.
        
        Given above information, we decided to vary the *'max_depth'* hyperparameter
        between 3 and 11, and the *min_child_weight* hyperparameter between 1 and 15. 
        The number of trees fixed at 500, and the remaining hyperparameters were set 
        as before.
                     
        The optimal combination of values was *'max_depth'* = 5 and *min_child_weight* = 5.
        The corresponding RMSE was 77.38.
        '''),
    html.Br(),
            
    # gamma, subsample and colsample_bytree  
    dcc.Markdown('''
        **Minimum loss reduction and subsample ratio**
        
        Using the previously defined hyperparameters, we tried to optimize the minimum 
        loss reduction required to make a further partition on a leaf node of the tree 
        (*gamma*), the *subsample* ratio of the training instances, and the subsample
        ratio of columns when constructing each tree (*colsample_bytree*). 
        
        We tested the following values for *'gamma' ranging from 0 to 100000. The 
        hyperparameter seemed to have litle influence on the RMSE, as only for 
        very large values did the RMSE increased. We decided to leave it
        set to 0. The re-calibration of the number of trees was not necessary, then.
        
        Nest, we tried combinations of values between 0.1 and 1.0 for the subsample 
        hyperparameters. The optimal values were *'subsample'* = 1 and *'colsample_bytree'* 
        = 1, which resulted in a RMSE value of 60.71.
        '''),
    html.Br(),
    
    # learning_rate
    dcc.Markdown('''
        **Maximum depth of a tree and minimum weight in a child**
        
        The XGBoost documentation informst that increasing the value of *'max_depth'*
        will make the model more complex and more likely to overfit. It also warns 
        that XGBoost aggressively consumes memory when training a deep tree.
        
        The minimum sum of instance weight needed in a child is defined by the 
        *'max_depth'* hyperparameter. According to the XGBoost documentation, the
        partitioning process will stop when the tree partition step results in a 
        leaf node with the sum of instance weight less than min_child_weight. 
        Thus, the larger its value is, the more conservative the algorithm will be.
        
        Given above information, we decided to vary the *'max_depth'* hyperparameter
        between 3 and 11, and the *min_child_weight* hyperparameter between 1 and 15. 
        The number of trees fixed at 500, and the remaining hyperparameters were set 
        as before.
                     
        The optimal combination of values was *'max_depth'* = 5 and *min_child_weight* = 5.
        The corresponding RMSE was 77.38.
        '''),
    html.Br(),
            
    dcc.Markdown('''
        **Learning rate**
        
        In the step of the tuning process, we tested lower values for learning rate,
        while at the same time increasing the number of trees. The 
        learning rate (or eta) is the step size shrinkage (error "correction") used
        to prevent overfitting. After each boosting step, it reduces the feature 
        weights to make the boosting process more conservative.
        
        We teste learning rate values between .01 and .75, for numbers of trees between
        100 and 5000. The optimal setting was *'learning_rate'* = .275 and 
        *'num_boost_round'* = 500. The corresponding RMSE was 55.59.
        '''),
    html.Br(),
    
    html.H4('Testing'),
    
    dcc.Markdown('''
        We used XGBoost with the previously defined hyperparameters
        to predict the number of cases of Dengue fever per in Belo Horizonte, 
        the capital of Minas Gerais state. As explained above, to obtain
        a unbiased estimate of the prediction error, we employed the
        rolling-origin-recalibration method with a
        period of 12 weeks as the testing window.
        
        The mean value for the RMSE obtained in the process was 627.18
        which is well below the values obtained with the models we 
        ran in the exploratory analysis. However, while the latter
        predicted values over a period of one year, the former predicted
        values for several twelve-week periods. Thus, the results should not
        be compared directly.
                                
        Because of this, we used the trained and validated model to predict the
        number of Dengue fever cases from May 05, 2018 to May 05,
        2019. The RMSE of the predicted values was 921.21, which is practically
        the same that we obtained in the exploratory analysis for the model with
        lagged outcomes for all cities and no time-related featues. 
        
        Below, we present the chart comparing the actual and predicted values for
        our final XGBoost model. As expected, it was very similar to that 
        generated for the last model tested in the exploratory analysis. The chart
        is followed by the importance plot for the 30 most important features
        in the final model.
        '''),
    
    # Chart: final model
    html.Div(
        className='row',
        children=[
            html.Img(src='data:image/png;base64,{}'.format(convert('pred-final-no-time-l4r26.png')),
                     width=900, height=450),
            html.Img(src='data:image/png;base64,{}'.format(convert('imp-final-no-time-l4r26.png')),
                     width=900, height=600)

        ]
    ),
    
    
    # html.Div(
    #     className='row',    
    #     children=[
    #         dcc.Graph(
    #             id='dist-importance',
    #             figure = {
    #                 'data': [
    #                     go.Scatter(
    #                         x=bh_imp['Distance'],
    #                         y=bh_imp['Importance'],
    #                         text=bh_imp['City'],
    #                         mode='markers',
    #                         opacity=0.5,
    #                         marker={
    #                             'size': 10,
    #                             'color': bh_imp['Lag'],
    #                             'colorscale': 'Bluered', 
    #                             'line': {'width': 0.5, 'color': 'white'}
    #                             }
    #                     )

    #                     #     x=bh_imp[bh_imp['Lag']==i]['Distance'],
    #                     #     y=bh_imp[bh_imp['Lag']==i]['Importance'],
    #                     #     text=bh_imp[bh_imp['Lag']==i]['City'],
    #                     #     mode='markers',
    #                     #     opacity=0.7,
    #                     #     marker={
    #                     #         'size': 10,
    #                     #         'color':  
    #                     #         'line': {'width': 0.5, 'color': 'white'}
    #                     #         },
    #                     # ) for i in bh_imp['Lag'].unique()
    #                 ],
    #                 'layout': go.Layout(
    #                             template='plotly_white',
    #                             title='Feature Importance against Distance',            
    #                             xaxis={'title': 'Distance'},
    #                             yaxis={'title': 'Log Importance'},
    #                             yaxis_type='log',
    #                             hovermode='closest'                                
    #                         )
    #                 }
    #         )
    #     ]   
    # )
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
            legend=dict(x=-.3, y=1),
            margin={'l':5, 'r':10, 'b':5, 't':20, 'pad':4}
        )
    }
    
    return corrs_figure
    
                
if __name__ == '__main__':
    app.run_server(debug=True)