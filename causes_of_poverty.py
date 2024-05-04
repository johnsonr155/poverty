import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl

from process_data import rename_raw_data, max_recent_yr

VARS_OF_INTEREST = [
    "Foreign direct investment, net inflows (BoP, current US$)",
    "Net official development assistance received (current US$)",
    "Access to electricity (% of population)",
    "Inflation, consumer prices (annual %)",
    "Individuals using the Internet (% of population)",
    "Armed forces personnel, total",
    "Agriculture, forestry, and fishing, value added (% of GDP)",
    "Manufacturing, value added (% of GDP)",
    "Services, value added (% of GDP)",
    "School enrollment, primary and secondary (gross), gender parity index (GPI)",
    "Unemployment, total (% of total labor force) (modeled ILO estimate)",
    "Refugee population by country or territory of origin",
    "Life expectancy at birth, total (years)",
    "Fertility rate, total (births per woman)",
    "Urban population (% of total population)",
    "GDP per capita (current US$)"
]


def relevant_metrics_data(df):
    #select relevant cols
    df = df[["country", "date", "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)"]+ VARS_OF_INTEREST]
    shortened_metric = {
        "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)":"poverty"
    }
    df.rename(columns=shortened_metric, inplace=True)
    return(df)

def filter_in_window_of_interest(df, gdp_range=[750, 2000]):
    df = df[(df['GDP per capita (current US$)'] > gdp_range[0]) & (df['GDP per capita (current US$)'] < gdp_range[1])]
    return(df)


def analyze_poverty_relationship_and_plot(df, dependent_var='poverty'):
    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Identifying predictors excluding 'country' and dependent variable
    predictors = [col for col in df.columns if col not in ['country', dependent_var]]
    
    results = []
    
    # Perform regression for each predictor
    for predictor in predictors:
        # Prepare the data by dropping NaNs only in relevant columns
        subset = df[[predictor, dependent_var]].dropna()

        # Normalize the predictor and dependent variable
        subset[predictor] = (subset[predictor] - subset[predictor].mean()) / subset[predictor].std()
        subset[dependent_var] = (subset[dependent_var] - subset[dependent_var].mean()) / subset[dependent_var].std()
        
        # Convert DataFrame to NumPy array for statsmodels and plotting
        X_np = subset[[predictor]].to_numpy()
        y_np = subset[dependent_var].to_numpy()
        
        # Add a constant to the independent variables for intercept
        X_np_with_const = sm.add_constant(X_np)
        
        # Fit the model
        model = sm.OLS(y_np, X_np_with_const).fit()
        
        # Store results
        coef = model.params[1]  # Get coefficient for the predictor
        r_squared = model.rsquared
        results.append({
            'Predictor': predictor,
            'Coefficient': coef,
            'R-squared': r_squared,
            'P-Value': model.pvalues[1]  # p-value for the predictor
        })

        # Plot if R-squared > 0.3
        if r_squared > 0.3:
            plt.figure(figsize=(10, 6))
            plt.scatter(X_np, y_np, color='blue', label='Data Points')
            plt.plot(X_np, model.predict(X_np_with_const), color='red', label=f'Regression Line\nR²={r_squared:.2f}')
            plt.title(f'Relationship between {predictor} and {dependent_var}\nCoefficient: {coef:.2f}')
            plt.xlabel(f'Normalized {predictor}')
            plt.ylabel(f'Normalized {dependent_var}')
            plt.legend()
            plt.grid(True)
            plt.show()

    # Creating a DataFrame to display the results
    results_df = pd.DataFrame(results)
    return results_df



df_renamed = rename_raw_data()
df_simplified = relevant_metrics_data(df_renamed)
max_year_poverty_by_country = max_recent_yr(df_simplified)
in_range_countries = filter_in_window_of_interest(df_simplified)

regression_table = analyze_poverty_relationship_and_plot(in_range_countries)
regression_table = regression_table[regression_table["R-squared"] > 0.3]

print(regression_table)