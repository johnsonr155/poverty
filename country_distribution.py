import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl

from process_data import rename_raw_data, max_recent_yr

GDP_RANGE = [740, 2000]

# select columns we're interested in for looking at disrribution
def simplify_data(df):
    #rename cols
    df = df[["country", "date", "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)", "GDP per capita (current US$)", "Population, total"]]
    shortened_metric = {
        "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)":"poverty",
        "GDP per capita (current US$)": "GDP per capita",
        "Population, total": "population"
    }
    df.rename(columns=shortened_metric, inplace=True)
    return(df)


# Plot bubble chart
# Customize axis tick labels 
def percent_formatter(x, pos):
    return f"{x:.0f}%"

def dollar_thousand_formatter(x, pos):
    return '${:,.0f}'.format(x)

def make_bubble_chart(df, gdp_range=GDP_RANGE):
    mpl.rcParams.update({'font.size': 14})

    fig = plt.figure(figsize=(7, 6))

    plt.scatter(df['GDP per capita'], df['poverty'], s=df['population'] / 1e6, alpha=0.5)
    plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_thousand_formatter))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    plt.axvspan(gdp_range[0], gdp_range[1], color='green', alpha=0.1)
    plt.text(1500, plt.ylim()[1]*0.9, 'Corridor of Variation', fontsize=10, ha='center')
    # Set labels and title
    plt.xlabel('GDP per capita')
    plt.ylabel('Percentage in poverty')
    plt.title('Poverty vs. GDP per Capita')
    # Show plot
    plt.grid(True)
    plt.show()
    fig.savefig('figs/distr_bubble.png', dpi=fig.dpi)


# focus plot
def make_focus_plot(df, gdp_range=GDP_RANGE):
    df = df[(df['GDP per capita'] > gdp_range[0]) & (df['GDP per capita'] < gdp_range[1])]
    fig = plt.figure(figsize=(7, 6))
    plt.scatter(df['GDP per capita'], df['poverty'], s=df['population'] / 1e5, alpha=0.5)
    plt.axvspan(gdp_range[0], gdp_range[1], color='green', alpha=0.1)
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(dollar_thousand_formatter))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    plt.xlim(gdp_range[0]-500, gdp_range[1]+500)
    plt.ylim(0, 70)
    # Set labels and title
    plt.xlabel('GDP per capita')
    plt.ylabel('Percentage in poverty')
    plt.title('Corridor of Variation: Poverty vs. GDP per Capita')
    for i, row in df.iterrows():
        plt.annotate(row['country'], (row['GDP per capita'], row['poverty']), textcoords="offset points", xytext=(5,5), ha='left', fontsize=11)
    # Show plot
    plt.grid(True)
    plt.show()
    fig.savefig('figs/distr_bubble_corridor.png', dpi=fig.dpi)



df_renamed = rename_raw_data()
df_simplified = simplify_data(df_renamed)
max_year_poverty_by_country = max_recent_yr(df_simplified)

make_bubble_chart(max_year_poverty_by_country)
make_focus_plot(max_year_poverty_by_country)