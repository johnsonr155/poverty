import pandas as pd


# -----------------ROUGH EXPLORATION OF DATA ----------------------


raw_data_filepath = "data/data_manager.csv"
indicator_ids_filepath =  "data/indicator_ids.csv"

df_raw = pd.read_csv(raw_data_filepath)
df_indicator = pd.read_csv(indicator_ids_filepath)

indicator_codes = set(df_raw.columns)

df_indicator_filtered = df_indicator[df_indicator['indicator_id'].isin(indicator_codes)]

code_to_metric = dict(zip(df_indicator_filtered['indicator_id'], df_indicator_filtered['indicator']))

df_raw.rename(columns=code_to_metric, inplace=True)

print(df_raw)

# Calculate completion of each column
completion = (1 - df_raw.isnull().mean()) * 100

# Print completion percentage for each column
print(completion)

# completion by country 
completion_by_country = df_raw.groupby('country')[['Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)']].apply(lambda x: (1 - x.isnull().mean()) * 100)
print(completion_by_country[completion_by_country["Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)"]>50])
