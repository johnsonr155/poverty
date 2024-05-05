import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl

from process_data import rename_raw_data, max_recent_yr

GDP_RANGE = [750, 2000]

VARS_OF_INTEREST = [
    "Access to electricity (% of population)",
    "Age dependency ratio (% of working-age population)",
    "Inflation, consumer prices (annual %)",
    "CO2 emissions (metric tons per capita)",
    "Individuals using the Internet (% of population)",
    "Agriculture, forestry, and fishing, value added (% of GDP)",
    "Manufacturing, value added (% of GDP)",
    "Services, value added (% of GDP)",
    "Mortality rate, infant (per 1,000 live births)",
    "School enrollment, primary and secondary (gross), gender parity index (GPI)",
    "Unemployment, total (% of total labor force) (modeled ILO estimate)",
    "Life expectancy at birth, total (years)",
    "Fertility rate, total (births per woman)",
    "Urban population (% of total population)",
    "GDP per capita (current US$)",
    "Ores and metals exports (% of merchandise exports)"
]

VARS_TO_ADJUST_PER_POP = [
    "Armed forces personnel, total",
    "Foreign direct investment, net inflows (BoP, current US$)",
    "Net official development assistance received (current US$)"
]

def relevant_metrics_data(df):
    #select relevant cols
    df = df[["country", "date", "Population, total", "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)"]+ VARS_OF_INTEREST + VARS_TO_ADJUST_PER_POP]
    df[VARS_TO_ADJUST_PER_POP] =  df[VARS_TO_ADJUST_PER_POP].div(df['Population, total'], axis=0)
    shortened_metric = {
        "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)":"poverty"
    }
    df.rename(columns=shortened_metric, inplace=True)
    return(df)

def filter_in_window_of_interest(df, gdp_range=GDP_RANGE):
    df = df[(df['GDP per capita (current US$)'] > gdp_range[0]) & (df['GDP per capita (current US$)'] < gdp_range[1])]
    return(df)

def percent_formatter(x, pos):
    return f"{x:.0f}%"

def analyze_poverty_relationship_and_plot(df, dependent_var='poverty', min_r2 = 0.01):
    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Specifying predictors excluding 'country' and dependent variable
    predictors = [col for col in df.columns if col not in ['country', dependent_var]]
    
    results = []
    # Perform regression for each predictor
    for predictor in predictors:
        # clean the data by dropping NaNs in relevant columns
        subset = df[[predictor, dependent_var]].dropna()

        # Normalize the predictor and dependent variable for the regression model
        norm_predictor = (subset[predictor] - subset[predictor].mean()) / subset[predictor].std()
        norm_dependent_var = (subset[dependent_var] - subset[dependent_var].mean()) / subset[dependent_var].std()
        
        # Convert to numpy arrays for regression and add constant for intercept
        X = sm.add_constant(norm_predictor.values.reshape(-1, 1))  
        y = norm_dependent_var.values
        
        # Fit the regression model
        model = sm.OLS(y, X).fit()
        
        # Store results
        coef = model.params[1] 
        r_squared = model.rsquared
        results.append({
            'Predictor': predictor,
            'Coefficient': coef,
            'R-squared': r_squared,
            'P-Value': model.pvalues[1]
        })

        # Plot if R-squared > above threshold using actual values
        if r_squared > min_r2:
            # Actual data for plotting
            actual_x = subset[predictor].values
            actual_y = subset[dependent_var].values
            predicted_y = subset[dependent_var].mean() + model.predict(X) * subset[dependent_var].std()

            mpl.rcParams.update({'font.size': 14})
            fig = plt.figure(figsize=(8, 5))
            plt.scatter(actual_x, actual_y, color='#008080', label='Countries in corridor (1990-)')
            plt.plot(actual_x, predicted_y, color='#836953', label=f'Regression Line\nR²={r_squared:.2f}')
            plt.title(f'{predictor} vs. Poverty')
            plt.xlabel(f'{predictor}')
            plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
            plt.ylabel(f'{dependent_var}')
            plt.legend()
            plt.grid(True)
            plt.show()
            fig.savefig(f'figs/{predictor} vs. Poverty.png', dpi=fig.dpi)


    # Creating df to display the results
    results_df = pd.DataFrame(results)
    return results_df


df_renamed = rename_raw_data()
df_simplified = relevant_metrics_data(df_renamed)
max_year_poverty_by_country = max_recent_yr(df_simplified)
in_range_countries = filter_in_window_of_interest(df_simplified)

regression_table = analyze_poverty_relationship_and_plot(in_range_countries)
regression_table = regression_table[regression_table["R-squared"] > 0.2]

print(regression_table)
