# Prepare the environment:
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen


## Scrape capitals, regions, UF

# Get the HTML code from the website and close browser
url = 'https://www.estadosecapitaisdobrasil.com/'
html = urlopen(url).read()

# Get table header with BeautifulSoup
soup = BeautifulSoup(html, 'lxml')
header = soup.find_all('th')

col_names = []

for col in header:
    col_names.append(col.get_text())

states = soup.find_all('tr')
cities = []

for row in states[1:len(states)]:
    city = []
    for col in row.find_all('td'):
        city.append(col.get_text().strip())
    cities.append(city)

df_cities = pd.DataFrame(cities, columns=col_names) \
    .drop(columns=['Bandeira','Curtir'])


## Scrape population from Wikipedia

# Get the HTML code from the website and close browser
url = 'https://pt.wikipedia.org/wiki/Lista_de_capitais_do_Brasil_por_popula%C3%A7%C3%A3o'
html = urlopen(url).read()

# Get all the table header with BeautifulSoup
soup = BeautifulSoup(html, 'lxml')
header = soup.find('table').find_all('th')

col_names = []

for col in header:
    col_names.append(col.get_text().strip())

states = soup.find('table').find_all('tr')
    
cities = []

for row in states[1:len(states)]:
    city = []
    for col in row.find_all('td'):
        city.append(col.get_text().strip().replace('.',''))
    cities.append(city)

df_cities_pop = pd.DataFrame(cities, columns=col_names) \
    .drop(columns=['Pos. 2018', 'Dif. 2000', 'Unidade federativa', 'Pos. 2000']) \
    .rename(columns={'Localidade':'municipio',
                     'População em 2018[1]':'pop_2018', 
                     'População em 2010':'pop_2010', 
                     'População em 2000':'pop_2000'})

df_cities_pop[['pop_2018', 'pop_2010', 'pop_2000']] = df_cities_pop[['pop_2018', 'pop_2010', 'pop_2000']] \
    .apply(pd.to_numeric)
    
df_cities = df_cities.merge(df_cities_pop, left_on='Capital', right_on='municipio') \
    .drop(columns='municipio') \
    .rename(columns={'Estado':'estado', 'Sigla':'UF', 'Capital':'municipio', 'Região':'regiao'})

df_cities.to_csv('Dados/capitais.csv', index=False)