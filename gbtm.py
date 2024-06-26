import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib as mpl
from process_data import read_and_rename_raw_data

# ----- Script for looking at individual country: and exploring its future poverty using GBTM of other countries in the past
# ************ Needs refactoring to work for general case of any country - currently hard coded for Uganda *****************

df_renamed = read_and_rename_raw_data()


# Quick look at Uganda
df_uganda = df_renamed[df_renamed["country"]=="Uganda"]
df_uganda = df_uganda[["country", "date", "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)", "GDP per capita (current US$)"]]
df_uganda = df_uganda.dropna()

print(df_uganda[df_uganda["date"]==2019])

GDP_RANGE = [750, 2000]

# Uganda's most recent poverty is 42% so looking either side
POVERTY_STARTING_POINT_RANGE=[37, 47]

def percent_formatter(x, pos):
    return f"{x:.0f}%"

# plotting Uganda's poverty trajectory to date
fig = plt.figure(figsize=(6, 2))
plt.plot(df_uganda['date'].values, df_uganda["Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)"].values, color='#008080')
plt.title('Uganda: Poverty trajectory to date')
plt.ylabel('Poverty')
plt.ylim(35, 80)
plt.xlim(1992, 2020)
plt.axvspan(1990, 2002, color='#800000', alpha=0.1)
plt.text(1995, plt.ylim()[1]*0.9, 'Phase 1', fontsize=10, ha='left')
plt.axvspan(2002, 2012, color='#008080', alpha=0.1)
plt.text(2005, plt.ylim()[1]*0.9, 'Phase 2', fontsize=10, ha='left')
plt.axvspan(2012, 2020, color='#800000', alpha=0.1)
plt.text(2013, plt.ylim()[1]*0.9, 'Phase 3', fontsize=10, ha='left')
plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
fig.savefig('figs/Uganda_past.png', dpi=fig.dpi)
plt.show()


def interpolate_and_mark(df):
    """function to interpolate between values for poverty and gdp per capita"""
    df = df.reset_index(drop=True)
    # Define function to interpolate and mark within each group
    def group_interpolate_and_mark(group):

        # Detect original NaNs for poverty and GDP - keep track of which values are interpolated and which were original
        poverty_nan_mask = group['poverty'].isna()
        gdp_nan_mask = group['GDP per capita (current US$)'].isna()

        # Interpolate the columns
        group['poverty'] = group['poverty'].interpolate()
        group['GDP per capita (current US$)'] = group['GDP per capita (current US$)'].interpolate()

        # Mark interpolated values
        group['poverty_interpolated'] = ((poverty_nan_mask & group['poverty'].notna())*1).astype(int)
        group['GDP_interpolated'] = ((gdp_nan_mask & group['GDP per capita (current US$)'].notna())*1).astype(int)
        return group

    # Apply the interpolation and marking to each group based on country
    df = df.groupby('country').apply(group_interpolate_and_mark)
    df = df.reset_index(drop=True)
    return(df)

def clean_data_for_trajectory_analysis(df):
    """function to clean, simplify and interpolate data"""
    shortened_metric = {
        "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)":"poverty"
    }
    df.rename(columns=shortened_metric, inplace=True)
    #select vars that we are interested in
    df = df[["date", "country", "poverty", "GDP per capita (current US$)"]]
    df = interpolate_and_mark(df)
    return(df)

# Finding countries in the GDP window that had a similar level of poverty to Uganda in the past
def countries_of_interest(df, gdp_range = GDP_RANGE, poverty_range=POVERTY_STARTING_POINT_RANGE, max_start_yr=2012):
    """function to find countries whose poverty and gdp per capita are in our range on interest and which have a long enough subsequent trajectory"""
    # rename poverty metric as poverty for simplicity
    df = df.dropna()
    # select countries that pass through the chosen range
    df = df[
        (df['GDP per capita (current US$)'] > gdp_range[0]) &
        (df['GDP per capita (current US$)'] < gdp_range[1]) &
        (df['poverty'] > poverty_range[0]) &
        (df['poverty'] < poverty_range[1]) &
        (df['date'] < max_start_yr)
        ]
    return(df["country"].unique())


# Extracting the historic trajectories for chosen countries 
def find_trajectories(df, countries, start_poverty_level = 42):
    """function that finds creates dataframe of trajectories based on a year 0 corresponding to the starting poverty level"""
    # Filter to include only the chosen countries
    df = df[df['country'].isin(countries)]
    # find year when trajectory passes closest to the chosen poverty level
    df['poverty_difference'] = abs(df['poverty'] - start_poverty_level)
    df.dropna(subset=['country', 'date', "poverty_difference"], inplace=True)
    # Find the year with the minimum absolute difference for each country
    min_poverty_indices = df.groupby('country')['poverty_difference'].idxmin()
    min_years = df.loc[min_poverty_indices, ['country', 'date']]

    # Create dictionary to map each country to its corresponding 'year 0'
    country_year0_map = dict(zip(min_years['country'], min_years['date']))
    # Calculate 'years_passed' column based on the difference between the current year and 'year 0' for each country
    df['years_passed'] = df.apply(lambda row: row['date'] - country_year0_map[row['country']], axis=1)

    return(df)


def gbtm(df, number_of_groups=3):
    """function to perform and plot group based trajectory modelling on countries in the same starting point as uganda"""
    df = df[df["years_passed"]>=0]
    df = df[df["years_passed"]<=10]

    #filter out countries with insufficient data points 
    real_poverty_counts = df[df['poverty_interpolated'] == 0].groupby('country')['poverty'].count()

    # Filter countries with at least 3 real poverty values
    countries_with_enough_data = real_poverty_counts[real_poverty_counts >= 2].index
    df = df[df['country'].isin(countries_with_enough_data)]

    pivot_data = df.pivot(index='country', columns='years_passed', values='poverty')

    # Replace NaN values with the subsequent value
    pivot_data = pivot_data.fillna(method='ffill', axis=1)

    # Scaling the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_data)

    # Fit a gaussian mixture model
    # The number of clusters or groups can be adjusted
    n_clusters = number_of_groups
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    gmm.fit(scaled_data)

    # Predict the cluster for each country
    clusters = gmm.predict(scaled_data)

    # Add the cluster labels to the dataframe
    pivot_data['Cluster'] = clusters

    original_data = scaler.inverse_transform(scaled_data)
    original_data_df = pd.DataFrame(original_data, index=pivot_data.index, columns=pivot_data.columns[:-1])

    # Add cluster information back to the dataframe with original data
    original_data_df['Cluster'] = clusters
    palette = ["#CC7722", "#800000", "#008080"]
    # Set up the plot
    mpl.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(8, 5))
    
    # Plot each country's poverty trajectory
    for index, row in original_data_df.iterrows():
        plt.plot(row.index[:-1].values, row.values[:-1], color=palette[int(row["Cluster"])])
        plt.plot(row.index[:-1].values, row.values[:-1], linewidth=40, color=palette[int(row["Cluster"])], alpha=0.2, label='_nolegend_')
        plt.text(row.index[-2], row.values[-2], index, fontsize=9, va='center', ha='left')
    
    # improve the plot
    plt.title('Past trajectories starting from ~40% in poverty')
    plt.xlabel('Years Passed')
    plt.ylabel('Poverty')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()
    # Define memorable names for each cluster
    patch1 = mpatches.Patch(color='#800000', label='Stuck in a rut')
    patch2 = mpatches.Patch(color='#CC7722', label='Steady progress')
    patch3 = mpatches.Patch(color='#008080', label='Economic miracle')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    plt.ylim(10, 50)
    plt.legend(handles=[patch1, patch2, patch3])
    plt.tight_layout()
    fig.savefig('figs/Uganda_GBTM.png', dpi=fig.dpi)
    plt.show()
    return(pivot_data)


df_renamed = read_and_rename_raw_data()
df_clean = clean_data_for_trajectory_analysis(df_renamed)

chosen_countries = countries_of_interest(df_clean)

df_trajectories = find_trajectories(df_clean, countries=chosen_countries)

df_test = gbtm(df_trajectories)
