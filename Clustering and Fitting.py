# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 20:46:06 2023

@author: cyber
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wbgapi as wb


indicator = ["EN.ATM.CO2E.PC","EG.USE.ELEC.KH.PC"]
country_code = ['AUS','CHN','CAN','FRA','RUS','NZL','DEU','USA','ARG']

def read_data(indicator,country_code):
    df = wb.data.DataFrame(indicator, country_code, mrv=30)
    return df

data= read_data(indicator, country_code)
print(data)
data.columns = [i.replace('YR','') for i in data.columns]
data=data.stack().unstack(level=1)
data.index.names = ['Country', 'Year']
print(data)
data = data.reset_index()
print(data)
data.drop(['EG.USE.ELEC.KH.PC'], axis = 1, inplace = True)
print(data)
data["Year"] = pd.to_numeric(data["Year"])
