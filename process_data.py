import pandas as pd

EXLCUDED_REGIONS = [
    "Africa Western and Central",
    "Africa Eastern and Southern",
    "Arab World",
    "Central Europe and the Baltics",
    "East Asia & Pacific (excluding high income)",
    "Early-demographic dividend",
    "East Asia & Pacific",
    "Europe & Central Asia (excluding high income)",
    "Europe & Central Asia",
    "Euro area",
    "European Union",
    "Fragile and conflict affected situations",
    "Heavily indebted poor countries (HIPC)",
    "IBRD only",
    "IDA only",
    "IDA & IBRD total",
    "IDA total",
    "IDA blend",
    "Latin America & Caribbean (excluding high income)",
    "Least developed countries: UN classification",
    "Low & middle income",
    "Late-demographic dividend",
    "Middle East & North Africa (excluding high income)",
    "Middle income",
    "OECD members",
    "Other small states",
    "Pre-demographic dividend",
    "Pacific island small states",
    "Post-demographic dividend",
    "Sub-Saharan Africa (excluding high income)",
    "Sub-Saharan Africa",
    "Small states",
    "East Asia & Pacific (IDA & IBRD countries)",
    "Europe & Central Asia (IDA & IBRD countries)",
    "Latin America & the Caribbean (IDA & IBRD countries)",
    "Middle East & North Africa (IDA & IBRD countries)",
    "South Asia (IDA & IBRD)",
    "Sub-Saharan Africa (IDA & IBRD countries)",
    "World",
    "High income",
    "Low income",
    "Lower middle income",
    "Upper middle income",
    "Not classified"] 

def rename_raw_data(raw_data_filepath = "data/data_manager.csv", indicator_ids_filepath = "data/indicator_ids.csv"):
    #read in data
    df_raw = pd.read_csv(raw_data_filepath)
    df_indicator = pd.read_csv(indicator_ids_filepath)
    df_raw = df_raw[~df_raw['country'].isin(EXLCUDED_REGIONS)]
    #rename columns
    indicator_codes = set(df_raw.columns)
    df_indicator_filtered = df_indicator[df_indicator['indicator_id'].isin(indicator_codes)]
    code_to_metric = dict(zip(df_indicator_filtered['indicator_id'], df_indicator_filtered['indicator']))
    df_raw.rename(columns=code_to_metric, inplace=True)
    return(df_raw)

# llooking only at the most recent data for each country 
def max_recent_yr(df, min_yr = 2015):
    df = df[df['poverty'].notnull()]
    df = df[df["date"] >= min_yr]
    df = df.groupby('country').apply(lambda x: x.loc[x['date'].idxmax()])
    df.reset_index(drop=True, inplace=True)
    return(df)

