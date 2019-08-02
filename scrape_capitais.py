# Prepare the environment:
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

# Open browser and go to website
driver = webdriver.Chrome()
driver.get('https://www.estadosecapitaisdobrasil.com/')

# Get the HTML code from the website and close browser
res = driver.execute_script("return document.documentElement.outerHTML")
driver.quit()

# Get all the references with BeautifulSoup
soup = BeautifulSoup(res, 'lxml')
states = soup.find_all('tr')

# Extract the title, authors and journal information
cities = []

for ref in states:
    ref_title = ref.find('a').text
    ref_grays = ref.find_all('div', class_='gs_gray')
    ref_authors = ref_grays[0].text.split(',')
    ref_authors = [author.strip() for author in ref_authors]
    ref_journal = ref_grays[1].text

    while '...' in ref_authors:
        ref_authors.remove('...')

    papers.append(
        {'title': ref_title, 'authors': ref_authors, 'journal': ref_journal})
