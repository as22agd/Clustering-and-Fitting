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
from scipy.optimize import curve_fit
import itertools as iter


# The list 'indicator' contains the required indicator IDs
indicator = ["EN.ATM.CO2E.PC","EG.USE.ELEC.KH.PC"]
# The list 'country_code' contains the codes of selected countries
country_code = ['AUS','BRA','CHN','DEU','GBR','IND']


# Funtion to read the data in world bank format to the dataframe
def read_data(indicator,country_code):
    """
    Parameters
    ----------
    indicator : list containing the required indicator IDs
    country_code : list containing the codes of selected countries
    Returns
    -------
    df : dataframe extracted from the dataset
    """
    df = wb.data.DataFrame(indicator, country_code, mrv=30)
    return df


# Function to normalise th data
def norm_df(df):
    """
    Parameters
    ----------
    df : dataframe with original values
    Returns
    -------
    df : normalised dataframe
    """
    y = df.iloc[:,2:]
    df.iloc[:,2:] = (y-y.min())/ (y.max() - y.min())
    return df


# Function for fitting
def fct(x, a, b, c):
    """
    
    Parameters
    ----------
    x : independent variable
    a,b,c : parameters to be fitted
    """
    return a*x**2+b*x+c


# Function to calculate the error ranges
def err_ranges(x, func, param, sigma):
    lower = func(x, *param)
    upper = lower
    uplow = []
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
    pmix = list(iter.product(*uplow))
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
    return lower, upper


# Reading data to dataframe using the function read_data(indicator,country_code)
data= read_data(indicator, country_code)

# Removing 'YR' and adding new index names to data
data.columns = [i.replace('YR','') for i in data.columns]
data=data.stack().unstack(level=1)
data.index.names = ['Country', 'Year']
data = data.reset_index()
data.drop(['EG.USE.ELEC.KH.PC'], axis = 1, inplace = True)
data["Year"] = pd.to_numeric(data["Year"])


# Normalised dataframe
dt_norm = norm_df(data)
df_fit = dt_norm.drop('Country', axis = 1)
# Applyying k-means clustering
k = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(df_fit)
# Plotting clusters of different countries based on CO2 emission
sns.scatterplot(data=dt_norm, x="Country", y="EN.ATM.CO2E.PC", hue=k.labels_)
plt.ylabel("CO2 Emission")
plt.title("CO2 Emission Rate ")
plt.legend()
plt.show()


# Dataframe containing the data of the country Australia
data1 = data[(data['Country'] == 'AUS')]
# Implementing curve_fit function
val = data1.values
x, y = val[:, 1], val[:, 2]
prmet, cov = opt.curve_fit(fct, x, y)

data1["pop_log"] = fct(x, *prmet)
print("Parameters are: ", prmet)
print("Covariance is: ", cov)
plt.plot(x, data1["pop_log"], label="Fit")
plt.plot(x, y, label="Data")
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.title("CO2 emission rate in Australia")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.show()

# Extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x, fct, prmet, sigma)

# Predicting the CO2 emission in next 10 years
low, up = err_ranges(2030, fct, prmet, sigma)
print("Forcasted CO2 emission in Australia 2030 ranges between", low, "and", up)

# Dataframe containing the data of the country China
data2 = data[(data['Country'] == 'CHN')]
# Implementing curve_fit function
val = data2.values
x, y = val[:, 1], val[:, 2]
prmet, cov = opt.curve_fit(fct, x, y)

data2["pop_log"] = fct(x, *prmet)
print("Parameters are: ", prmet)
print("Covariance is: ", cov)
plt.plot(x, data2["pop_log"], label="Fit")
plt.plot(x, y, label="Data")
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.title("CO2 emission rate in China")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.show()

# Extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x, fct, prmet, sigma)

# Predicting the CO2 emission in next 10 years
low, up = err_ranges(2030, fct, prmet, sigma)
print("Forcasted CO2 emission in China in 2030 ranges between", low, "and", up)

# Dataframe containing the data of the country United Kingdom
data3 = data[(data['Country'] == 'GBR')]
# Implementing curve_fit function
val = data3.values
x, y = val[:, 1], val[:, 2]
prmet, cov = opt.curve_fit(fct, x, y)

data3["pop_log"] = fct(x, *prmet)
print("Parameters are: ", prmet)
print("Covariance is: ", cov)
plt.plot(x, data3["pop_log"], label="Fit")
plt.plot(x, y, label="Data")
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.title("CO2 emission rate in United Kingdom")
plt.legend(loc='best', fancybox=True, shadow=True)
plt.show()

# Extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(cov))
print(sigma)
low, up = err_ranges(x, fct, prmet, sigma)

# Predicting the CO2 emission in next 10 years
low, up = err_ranges(2030, fct, prmet, sigma)
print("Forcasted CO2 emission in United Kingdom in 2030 ranges between", low, "and", up)