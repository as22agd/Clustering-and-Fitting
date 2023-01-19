# -*- coding: utf-8 -*-
"""

Program to perform clustering and fitting in given datasets

"""
# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wbgapi as wb
from sklearn.cluster import KMeans
import seaborn as sns
import scipy.optimize as opt

# The list 'indicator' contains the required indicator IDs
indicator = ["EN.ATM.CO2E.PC","EG.USE.ELEC.KH.PC"]
# The list 'country_code' contains the codes of selected countries
country_code = ['AUS','BRA','CHN','DEU','GBR','IND']


# Funtion to read the data in world bank format to the dataframe
def read_data(indicator,country_code):
    df = wb.data.DataFrame(indicator, country_code, mrv=30)
    return df


# Function to normalise th data
def norm_df(df):
    y = df.iloc[:,2:]
    df.iloc[:,2:] = (y-y.min())/ (y.max() - y.min())
    return df


# Reading data to dataframe using the function read_data(indicator,country_code)
data= read_data(indicator, country_code)

# Removing 'YR' and adding new index names to data
data.columns = [i.replace('YR','') for i in data.columns]
data=data.stack().unstack(level=1)
data.index.names = ['Country', 'Year']
data = data.reset_index()
data.drop(['EG.USE.ELEC.KH.PC'], axis = 1, inplace = True)
data["Year"] = pd.to_numeric(data["Year"])
data = data.reset_index()

# Normalised dataframe
dt_norm = norm_df(data)
df_fit = dt_norm.drop('Country', axis = 1)
# Applyying k-means clustering
k = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(df_fit)
# Plotting clusters of different countries based on CO2 emission
sns.scatterplot(data=dt_norm, x="Country", y="EN.ATM.CO2E.PC", hue=k.labels_)
plt.legend()
plt.show()
