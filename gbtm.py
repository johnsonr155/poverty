import pandas as pd

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl

from process_data import rename_raw_data, max_recent_yr

def countries_of_interest(df, gdp_range = [500, 2000], poverty_range=[20, 30], max_start_yr=2005):
    """We only want to look at countries whose poverty and gdp per capita in the 90s are in our range on interst"""
    shortened_metric = {
        "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)":"poverty"
    }
    df.rename(columns=shortened_metric, inplace=True)
    df = df[
        (df['GDP per capita (current US$)'] > gdp_range[0]) &
        (df['GDP per capita (current US$)'] < gdp_range[1]) &
        (df['poverty'] > poverty_range[0]) &
        (df['poverty'] < poverty_range[1]) &
        (df['date'] < max_start_yr)
        ]
    return(df["country"].unique())

df_renamed = rename_raw_data()
CHOSEN_COUNTRIES = countries_of_interest(df_renamed)

#df_simplified = relevant_metrics_data(df_renamed)
#max_year_poverty_by_country = max_recent_yr(df_simplified)
#in_range_countries = filter_in_window_of_interest(df_simplified)

def plot_trajectories(df, countries = CHOSEN_COUNTRIES):
    # Filter the DataFrame to include only the chosen country names
    df = df[df['country'].isin(countries)]
    df = df[["date", "country", "poverty"]]

    df['poverty_difference'] = abs(df['poverty'] - 25)
    df.dropna(subset=['country', 'date', "poverty_difference"], inplace=True)
    # Find the year with the minimum absolute difference for each country
    min_poverty_indices = df.groupby('country')['poverty_difference'].idxmin()

    min_years = df.loc[min_poverty_indices, ['country', 'date']]

    # Create dictionary to map each country to its corresponding 'year 0'
    country_year0_map = dict(zip(min_years['country'], min_years['date']))
    # Calculate 'years_passed' column based on the difference between the current year and 'year 0' for each country
    
    df['years_passed'] = df.apply(lambda row: row['date'] - country_year0_map[row['country']], axis=1)
    print(df)

    # Plot line chart for each country
    plt.figure(figsize=(10, 6))
    for country in df.country.unique():
        data = df[df["country"]==country]
        plt.plot(data['years_passed'].values, data['poverty'].values, color='#008080')

    # Set labels and title
    plt.axvspan(-5, 0, color='gray', alpha=0.3)

    plt.xlabel('years_passed')
    plt.ylabel('Poverty')
    plt.title('Poverty Over Time for Chosen Countries')
    plt.xlim(-5, 30)
    plt.ylim(0, 40)
    # Add legend
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()

plot_trajectories(df_renamed)