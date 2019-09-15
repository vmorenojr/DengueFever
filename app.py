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

# Read XGBoost results
bh_lags = pd.read_csv('XGB_fit/baseline-lags.csv')
bh_caps = pd.read_csv('XGB_fit/allcaps.csv')


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
                    value=['Centro-Oeste'],
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
        - RMSE: 1839.45
        - MAE (Mean Absolute Error): 773.34
        - MAPE (Mean Absolute Percentage Error): 146.88
        '''),
    html.Br(),
    
    dcc.Markdown('''    
        Of the time-related predictors, year, day of the year and week of the year were by far the 
        most important. This may be due to the rainy and dry seasons cycles not conforming exactly to 
        months, thereby making the number of the month in a year and the day of the month less 
        relevant to prediction.  
        '''),
    
    # Charts
    html.Div(
        className='row',
        children=[
            html.Div(
                className='five columns',
                children=[
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
        Next, we examined models with lagged data on the occurrence of Dengue in Belo Horizonte. 
        Different lags and ranges of lagged ocurrences were tested. The following table
        summarizes the fit results.             
        '''),
    
    # Table: Fit results for Belo Horizonte
    html.Div(
        className='row',
        children=[
            html.Div(
                className='six columns',
                children=generate_table(bh_lags.drop('All cities', axis=1)\
                                               .round(decimals=2), max_rows=20)
            ),
            html.Div(
                className='six columns',
                children=[
                    html.Br(),
                    dcc.Markdown('''    
                        When compared with the baseline results, only those for a lag of four weeks showed 
                        substantial improvements. It is interesting to note that the best results for RMSE 
                        and MAPE were different. This applies to the results for all lags.               
                    ''')
                ]
            )
        ]
    ),
    html.Br(),
    
    dcc.Markdown('''    
        The following charts compare the actual Dengue fever occurrences with
        the predicted values. We selected those corresponding to the best RMSE 
        result for each lag.
        '''),
    html.Br(),
    
    # Charts: Belo Horizonte with lags
    html.Div(
        className='row',
        children=[
            html.Div(
                className='six columns',
                children=[
                    dcc.Markdown('''**Lag: 4 weeks, Range: 1 year**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('pred-base-l4r52.png')),
                             width=600, height=300),
                    
                    dcc.Markdown('''**Lag: 6 months, Range: 12 weeks**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('pred-base-l26r12.png')),
                             width=600, height=300)
                ]
            ),
            html.Div(
                className='six columns',
                children=[
                    dcc.Markdown('''**Lag: 12 weeks, Range: 1 year**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('pred-base-l12r52.png')),
                             width=600, height=300),
                    
                    dcc.Markdown('''**Lag: 1 year, Range: 12 weeks**'''),
                    html.Img(src='data:image/png;base64,{}'.format(convert('pred-base-l52r12.png')),
                             width=600, height=300)
                ]
            )
        ]
    ),
    html.Br(),   
    
    
    # html.Div(
    #     className='row',
    #     children=[
    #         html.Div(
    #             className='five columns',
    #             children=[
    #                 html.Img(src='data:image/png;base64,{}'.format(convert('ts-baseline.png')),
    #                          width=500, height=250),
    #                 html.Img(src='data:image/png;base64,{}'.format(convert('pred-base.png')),
    #                          width=500, height=250)
    #             ]
    #         ),
    #         html.Div(
    #             className='seven columns',
    #             children=
    #                 html.Img(src='data:image/png;base64,{}'.format(convert('imp-base.png')),
    #                          width=780, height=520)
    #         )
    #     ]
    # ),
    
    
    
    
    
    
    
    
    
    
    
    
    dcc.Markdown('''                  
        The following steps were used to train and define the hyperparameters (
        [XGBoost, 2019](https://xgboost.readthedocs.io/en/latest/parameter.html);
        [Jain, 2016](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/);
        [Cambridge Spark, 2017](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f)):
        
        First, we chose a learning rate of 0.1 and used xgboost *cv* function to obtain
        the optimum number of trees. XGBoost's *cv* performs cross-validation at each boosting
        iteration and returns the optimum number of trees. We set the number of folds to five.
                        
        Second, we tuned two of the most important tree-specific parameters, *max_depth* and
        *min_child_weight* for the selected learning rate and the number of trees found in the
        previous step. We used scikit learn's *GridSearchCV* with five folds for this purpose.
                        
        Third, we tried to reduce model complexity and enhance performance by tuning the
        parameters *gamma*, *subsample*, and *colsample_bytree*.
                            
        Finally, we lowered the learning rate to define the optimal set of parameters.
                        
        Our findings are reported in more detail below.
    ''')
    
    # html.H6('Number of trees'),
        
    # html.Div(
    #     className='row',    
    #     children=[
    #         html.Div(
    #             className='six columns',
    #             children=
    #                 dcc.Markdown('''
    #                     We followed the guidelines obtained in the above-mentioned documents
    #                     and defined the following initial values for our parameters:
    #                     - __*learning_rate*__: .1
    #                     - __*min_child_weight*__: 1          
    #                     - __*max_depth*__: 6
    #                     - __*gamma*__: 0
    #                     - __*subsample*__: 1
                        
    #                     In XGBoost's *cv*, we set maximum number of boosting rounds to 1000, with an early 
    #                     stopping at 50 rounds. The results are shown in the chart to the
    #                     right.
                        
    #                     We compromised between performance, the availability
    #                     computational resources and the possibility of overfitting, 
    #                     deciding to set the number of trees to 25. The corresponding RMSE was 107.9.
    #                 ''')
    #         ),
                       
    #         html.Div(
    #             className='six columns',
    #             children=
    #                 dcc.Graph(
    #                     id='xg_trees',
    #                     figure = {
    #                         'data': [
    #                             go.Scatter(
    #                                 x = xgb_trees['Trees'],
    #                                 y = xgb_trees['Mean RMSE'],
    #                                 mode = 'lines',
    #                                 line=dict(color='firebrick'),
    #                                 showlegend = False
    #                             )],
    #                         'layout': go.Layout(
    #                                 template='plotly_white',
    #                                 title= 'Mean RMSE by Number of Trees in the Model',            
    #                                 xaxis={'title': 'Number of Trees'},
    #                                 yaxis={'title': 'Mean RMSE'},
    #                                 hovermode='closest',
    #                                 margin={'l':30, 'r':30, 'b':10, 't':30}
    #                                 )
    #                         }
    #                 )
    #         )
    #     ]
    # ),
    # html.Br(),
    
    # html.H6('Tuning max_depth and min_child_weight'),
    
    # dcc.Markdown('''
    #     We used scikit-learn's GridSearchCV to optimize the maximum depth
    #     and the minimum sum of instance weight (hessian) needed in a child.
    #     We used the following values:
    #     - __*min_child_weight*__: .1, .5, 1, 2, 5          
    #     - __*max_depth*__: 4, 6, 8, 10
        
    #     The optimal settings were *'max_depth'* = 4, and *min_child_weight* = 2.
        
    #     Given the importance of those hyperparameters, we tried to refine 
    #     the optimal results by varying them with smaller increments. The 
    #     following values were tested in the second round:
    #     - __*min_child_weight*__: 1.5, 2, 2.5       
    #     - __*max_depth*__: 3, 4, 5
        
    #     The optimal values were *'max_depth'* = 5, and *min_child_weight* = 2.5.
    #     '''
    # ),
    # html.Br(),
            
    # html.H6('Tuning gamma, subsample and colsample_bytree'),

    # dcc.Markdown('''
    #     Using the previously defined hyperparameters, we used again scikit-learn's 
    #     GridSearchCV to try to optimize the minimum loss reduction required to make a further
    #     partition on a leaf node of the tree (*gamma*), the *subsample* ratio of the
    #     training instances, and the subsample ratio of columns when constructing each 
    #     tree (*colsample_bytree*). We tested the following values:
    #     - __*gamma*__: 0, .1, .25, .5, 1, 2          
    #     - __*subsample*__: .3, .6, 1
    #     - __*colsample_bytree*__: .3, .6, 1
                        
    #     The optimal settings were *'gamma'* = 0, *'subsample'* = 1, and *'colsample_bytree'* = 1.
    #     '''
    # ),
    # html.Br(),
    
    # html.H6('Tuning learning_rate'),
            
    # html.Div(
    #     className='row',    
    #     children=[
    #         html.Div(
    #             className='six columns',
    #             children=[
    #                 dcc.Markdown('''
    #                     In a final step, we return to XGBoost's *cv* to test a lower learning rate,
    #                     while at the same time allowing the number of trees to increase. The 
    #                     learning rate (or eta) is the step size shrinkage (error "correction") used
    #                     to prevent overfitting. After each boosting step, it reduces the feature 
    #                     weights to make the boosting process more conservative.
                        
    #                     The following settings were used:
    #                     - __*learning_rate*__: .01 
    #                     - __*num_boost_round*__: 5000
    #                     '''),
    #                 html.Br(),
                    
    #                 dcc.Markdown('''
    #                     Based on the chart to the right, the best number of estimators (trees) would
    #                     be around 250. The corresponding RMSE was 108.6.
    #                     ''')
    #             ]
    #         ),
    #         html.Div(
    #             className='six columns',
    #             children=
    #                 dcc.Graph(
    #                     id='xg_learning',
    #                     figure = {
    #                         'data': [
    #                             go.Scatter(
    #                                 x = xgb_learning['Trees'],
    #                                 y = xgb_learning['Mean RMSE'],
    #                                 mode = 'lines',
    #                                 line=dict(color='firebrick'),
    #                                 showlegend = False
    #                             )],
    #                         'layout': go.Layout(
    #                                 template='plotly_white',
    #                                 title= 'Mean RMSE by Number of Trees in the Model',            
    #                                 xaxis={'title': 'Number of Trees'},
    #                                 yaxis={'title': 'Mean RMSE'},
    #                                 hovermode='closest',
    #                                 margin={'l':30, 'r':30, 'b':10, 't':30}
    #                                 )
    #                         }
    #                 )
    #         )
    #     ]
    # ),
    # html.Br(),
    
    # html.H4('Testing'),
    
    # html.Div(
    #     className='row',    
    #     children=[
    #         html.Div(
    #             className='six columns',
    #             children=
    #                 dcc.Markdown('''
    #                     We used XGBoost with the previously defined hyperparameters
    #                     to predict the number of cases of Dengue fever per 100,000k
    #                     population in Belo Horizonte, the capital of Minas Gerais state.
                        
    #                     The RMSE of the predicted values was 108.5, which is quite similar
    #                     to what was obtained in the training and validation stages. This
    #                     RMSE is quite high, and similar to the standard deviation of the
    #                     target feature in the original dataset.
                        
    #                     The chart to the right confirms that the predicted values 
    #                     (in red) were quite off the observed values (in blue). Thus,
    #                     our model had a poor predictive power.
    #                     ''')
                    
    #         ),
    #         html.Div(
    #             className='six columns',
    #             children=
    #                 dcc.Graph(
    #                     id='xg_test',
    #                     figure = {
    #                         'data': [
    #                             go.Scatter(
    #                                 x = xgb_test['data_alvo'],
    #                                 y = xgb_test['por_habitante_pred'],
    #                                 mode = 'lines',
    #                                 line=dict(color='firebrick'),
    #                                 showlegend = False
    #                             ),
    #                             go.Scatter(
    #                                 x = xgb_test['data_alvo'],
    #                                 y = xgb_test['por_habitante_alvo'],
    #                                 mode = 'lines',
    #                                 line=dict(color='darkblue'),
    #                                 showlegend = False
    #                             )],
    #                         'layout': go.Layout(
    #                                 template='plotly_white',
    #                                 title= 'Mean RMSE by Number of Trees in the Model',            
    #                                 xaxis={'title': 'Number of Trees'},
    #                                 yaxis={'title': 'Mean RMSE'},
    #                                 hovermode='closest',
    #                                 margin={'l':30, 'r':30, 'b':10, 't':30}
    #                                 )
    #                         }
    #                 )
    #         )
    #     ]
    # ),
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