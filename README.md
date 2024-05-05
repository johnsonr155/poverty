# Understanding Global Poverty 

## Overview
This repo  explores factors contributing to poverty across various countries. It tries to identify factors causing poverty and looks at past trajectories of poverty for different countries as a means for thinking about the future

This repo is a WIP and needs further refactoring

## Features
- **Data Processing**: Functions for reading in the raw data and doing some processing
- **Exploratory Data Analysis**: very rough script used for inital scoping analysis
- **Poverty Distribution Analysis**: Investigates the distribution of poverty and the relationship with gdp per capita
- **Predictor Analysis**: Regression analyssi of variables that could contribute to poverty levels. (Based on countries within a narrow range of GDP per capita)
- **Group Based Trajectory Modeling**: Uses a GBTM to cluster histroic poverty trajectories with a view to thinking about possible futures


### Prerequisites
- Python 3.6 or higher
- Required Python libraries: Pandas, Numpy, Matplotlib, Statsmodels

### Installation
Clone the repository and install the necessary Python packages:

```bash
git clone https://github.com/johnsonr155/poverty.git
cd poverty
pip install -r requirements.txt
