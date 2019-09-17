## Predicting Dengue Fever Epidemics in Brazilian State Capitals

### Introduction

This project is part of the InfoDengue project developed by Prof. Flavio Codeço and colleagues from Fundação Getúlio Vargas (FGV) and Fundação Oswaldo Cruz (Fiocruz). In addition to its various research-oriented deliveries, the project has implemented an online dashboard that is weekly updated with Dengue, Chikungunya and Zika fever data from 790 cities in Brazil. In this way, it enables local health agencies to make better decisions as far as planning and the allocation of resources is concerned, and subsenquently evaluate the outcomes of its efforts.

Our project focuses on the dynamics of Dengue fever epidemics in the 27 Brazilian state capitals. Its main objective is to assess the possibility of using the distances among those cities and the historical data on Dengue fever reports to predict future epidemics. Different machine learning (ML) techniques were used to make the predictions. Their accuracy results were compared and assessed, and the practical implications of our findings, discussed.

The web site of the project, with its description, ancillary videos, interactive charts, findings, and conclusions is at https://denguebr.herokuapp.com.

### Installation

To run all modules in our project, follow these steps:

1. Clone the github repository to a folder in your computer.
2. Open a PowerShell in that folder, and create and activate a virtual environment. If necessary, detailed instructions can be found [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
3. To install all required modules, run: `$pip install -r requirements.txt` 
4. It may be necessary to install xgboost manually. If so, you can find detailed instructions [here](https://xgboost.readthedocs.io/en/latest/build.html)

### Main application

The main application (app.py) consists of an interactive website with information on the project, the steps we took in its implementation, and our findings and conclusions.

To run the main application, type `$py app.py` in a PowerShell opened within the project folder. Then, in your web browser, go to the address that is show in the PowerShell (ex., 127.0.0.1:8050).

### Modules

We developed individual python modules for a variety of tasks. They usually generate results that are recorded as csv files. These intermediary results are read into pandas dataframes in other modules.

Here is a list of the individual modules of the project:
  - *app.py*: presents the project report in a web site
  - *cross-corr.py*: calculates the cross-correlations between time series of Dengue fever occurrences
  - *distance_matrix.py*: accesses Google Maps to get the distance between state capitals
  - *gen_datasets.py*: creates datasets with Dengue occurrences for each state capital
  - *IBGE.py*: set of ancillary functions for data wrangling
  - *per_capta.py*: interpolates population data to get estimates of each city population over the time period used in our analyses
  - *scrape_capitals*: scrapes additional data on state capitals from web sites
  - *wrangling.py*: prepares a clean file with all the information received from the InfoDengue project
  - *xgb_model.py*: runs the XGBoost analyses

### Jupyter Notebooks

All modules of the project are also available as Jupyter notebooks. They are within the Notebooks folder, which is in the main project folder.

Please note that charts are interactive only in the project website.

### Additional folders

The following folders will be created when cloning the github repository:
  - Dados: contains all datasets and files with intermediary results generated by the ancillary modules
  - Notebooks: contains the Jupyter notebooks for the project
  - Plots: contains the PNG files for the plots generated in the XGBoost analyses
  - XGB_fit: contains files with the fit results generated in the XGBoost analyses

### Contact

If you have any questions or suggestions, please contact the authors:
  - Lucas Oliveira:
  - Valter Moreno: valter.moreno@eng.uerj.br

