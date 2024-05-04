import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

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
    print(len(df["country"].unique()))

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
    return(df)

df_trajectories = plot_trajectories(df_renamed)
print(df_trajectories)

def gbtm(df):
    df = df[df["years_passed"]>-1]
    pivot_data = df.pivot(index='country', columns='years_passed', values='poverty')

    # Replace any possible NaN values with the mean of the column (optional based on your data)
    pivot_data = pivot_data.fillna(method='ffill', axis=1)

    # Scaling the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_data)

    # Fit a Gaussian Mixture Model
    n_components = 3  # The number of trajectories you hypothesize
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(scaled_data)

    # Predict the cluster for each country
    clusters = gmm.predict(scaled_data)

    # Add the cluster labels to the DataFrame
    pivot_data['Cluster'] = clusters

    print(pivot_data)
    original_data = scaler.inverse_transform(scaled_data)
    original_data_df = pd.DataFrame(original_data, index=pivot_data.index, columns=pivot_data.columns[:-1])

    # Add cluster information back to the DataFrame with original data
    original_data_df['Cluster'] = clusters
    palette = ["#008080", "#6C22A6", "#CC7722"]
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Plot each country's poverty trajectory
    for _, row in original_data_df.iterrows():
        plt.plot(row.index[:-1].values, row.values[:-1], color=palette[int(row["Cluster"])])
        plt.plot(row.index[:-1].values, row.values[:-1], linewidth=40, color=palette[int(row["Cluster"])], alpha=0.2, label='_nolegend_')


    # Enhance the plot
    plt.title('Poverty Trajectories by Country Cluster')
    plt.xlabel('Years Passed')
    plt.ylabel('Poverty Level')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()
    patch1 = mpatches.Patch(color='#6C22A6', label='Stuck in a rut')
    patch2 = mpatches.Patch(color='#008080', label='On the right track')
    patch3 = mpatches.Patch(color='#CC7722', label='Rags to riches')

    plt.legend(handles=[patch1, patch2, patch3])
    # Show the plot
    plt.tight_layout()
    plt.show()
    return(pivot_data)

df_test = gbtm(df_trajectories)
print(df_test)