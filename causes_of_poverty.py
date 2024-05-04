import pandas as pd
import numpy as np
import statsmodels.api as sm

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

 
def analyze_poverty_relationship(df, dependent_var='poverty'):
    # Normalize the data first (if there are any infinities, replace them with NaN first)
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
        
        # Adding a constant for regression intercept
        X = sm.add_constant(subset[[predictor]])
        y = subset[dependent_var]
        
        # Fit the model
        model = sm.OLS(y, X).fit()
        
        # Store results
        results.append({
            'Predictor': predictor,
            'Coefficient': model.params[predictor],
            'R-squared': model.rsquared,
            'P-Value': model.pvalues[predictor]
        })

    # Creating a DataFrame to display the results
    results_df = pd.DataFrame(results)
    return results_df




df_renamed = rename_raw_data()
df_simplified = relevant_metrics_data(df_renamed)
max_year_poverty_by_country = max_recent_yr(df_simplified)
in_range_countries = filter_in_window_of_interest(df_simplified)

regression_table = analyze_poverty_relationship(in_range_countries)
regression_table = regression_table[regression_table["R-squared"] > 0.3]

print(regression_table)