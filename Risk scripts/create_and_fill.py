import configparser
import datetime
from datetime import timedelta, date
from TM1py.Objects import Cube, Dimension, Hierarchy, Element, ElementAttribute
from TM1py.Services import TM1Service
import numpy as np
from pandas import read_csv
import pandas as pd
from numpy.fft import *
import sys
from forex_python.converter import CurrencyRates

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

config = configparser.ConfigParser()
config.read('D:/TM1 Models/AnalogicExternalDrivers/Python work/process_scripts/tm1conf.ini', encoding='utf-8')
tm1 = TM1Service(**config['Analogic External Drivers'])
    
#Create Set dimension
currency_set = ('Live','Base','EXP','HOLT','ARIMA')
elements = [Element(cur, 'Numeric') for cur in currency_set]
hierarchy = Hierarchy('zSYS Analogic External Drivers Currency Set', 'zSYS Analogic External Drivers Currency Set', elements)
dimension = Dimension('zSYS Analogic External Drivers Currency Set', [hierarchy])

if tm1.dimensions.exists(dimension.name):
    tm1.dimensions.delete(dimension.name)
    
if not tm1.dimensions.exists(dimension.name):  
    tm1.dimensions.create(dimension)
    
#Create Measure dimension
measure_modes = ('Train','Test','Simple','Rolling')
elements = [Element(measure, 'Numeric') for measure in measure_modes]
hierarchy = Hierarchy('zSYS Analogic External Drivers Currency Measure', 'zSYS Analogic External Drivers Currency Measure', elements)
dimension = Dimension('zSYS Analogic External Drivers Currency Measure', [hierarchy])

if tm1.dimensions.exists(dimension.name):
    tm1.dimensions.delete(dimension.name)
if not tm1.dimensions.exists(dimension.name):  
    tm1.dimensions.create(dimension)
    

start_date = sys.argv[1]
end_date = sys.argv[2]
base_currency = sys.argv[3]
currencies = sys.argv[4]

c = CurrencyRates()

start_date = datetime.datetime.strptime(start_date, "%Y.%m.%d")
start_date = start_date.date()
end_date = datetime.datetime.strptime(end_date, "%Y.%m.%d")
end_date = end_date.date()

cube_start_date = '2010.01.01'
cube_start_date = datetime.datetime.strptime(cube_start_date, "%Y.%m.%d")
cube_start_date = cube_start_date.date()
    
##Create Calendar Day dimension
#elements = [Element(str(single_date), 'Numeric') for single_date in daterange(cube_start_date, end_date)]
#hierarchy = Hierarchy('Currency Calendar Day', 'Currency Calendar Day', elements)
#dimension = Dimension('Currency Calendar Day', [hierarchy])
#if tm1.dimensions.exists(dimension.name):
#    tm1.dimensions.delete(dimension.name)
#if not tm1.dimensions.exists(dimension.name):
#    tm1.dimensions.create(dimension)

#Create Currency dimension
    
currencies = currencies.split(",")
elements = [Element(currency, 'Numeric') for currency in currencies]
hierarchy = Hierarchy('Currency', 'Currency', elements)
dimension = Dimension('Currency', [hierarchy])
if tm1.dimensions.exists(dimension.name):
    tm1.dimensions.delete(dimension.name)
if not tm1.dimensions.exists(dimension.name):  
    tm1.dimensions.create(dimension)
    

#Create Currency cube
    
cube = Cube('zSYS Analogic External Drivers Currency', ['zSYS Analogic External Drivers Currency Set', 'Currency Calendar Day', 'Currency','zSYS Analogic External Drivers Currency Measure'])
tm1.cubes.create(cube)


#Create pandas DF    
days_num = end_date-start_date
days_num = days_num.days

currency_df = pd.DataFrame()

for i in range(0,days_num):

    day = start_date+datetime.timedelta(days=i)

    curr_rates = c.get_rates(base_currency, day)  
    curr_rates = pd.DataFrame.from_dict(curr_rates,orient='index', columns=[str(day)])
    currency_df = pd.concat([currency_df, curr_rates], axis=1, sort=False)
    
curr_ind = currency_df.index
del_list = list(set(curr_ind)-set(list(currencies)))
currency_df = currency_df.drop(labels = del_list, axis = 0)

#Insert data into cube      
cellset = {}
for curr in currency_df.index:
    for date in currency_df.columns:
        
        cellset[('Base', str(date), curr, 'Train')] = currency_df.loc[curr][date]

tm1.cubes.cells.write_values(cube.name, cellset)