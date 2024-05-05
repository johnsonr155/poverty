import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

from process_data import rename_raw_data

df_renamed = rename_raw_data()

df_cameroon = df_renamed[df_renamed["country"]=="Cameroon"]
df_cameroon = df_cameroon[["country", "date", "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)"]]
print(df_cameroon)
