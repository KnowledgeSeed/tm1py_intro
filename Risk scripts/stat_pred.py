import configparser
from datetime import timedelta, date
from TM1py.Objects import Cube, Dimension, Hierarchy, Element, ElementAttribute
from TM1py.Services import TM1Service
import numpy as np
from pandas import read_csv
import pandas as pd
from numpy.fft import *
import sys
from forex_python.converter import CurrencyRates
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

config = configparser.ConfigParser()
config.read('D:/TM1 Models/AnalogicExternalDrivers/Python work/process_scripts/tm1conf.ini', encoding='utf-8')
tm1 = TM1Service(**config['Analogic External Drivers'])
    
currency = pd.read_excel('D:/TM1 Models/AnalogicExternalDrivers/Python work/process_scripts/Currency.xlsx')
currency.columns = ['Date','CHF']
currency = currency.set_index(pd.DatetimeIndex(currency['Date'])) 
currency = currency.CHF
currency = currency[currency.index.day.isin([28])]

X = currency.values


stat = sys.argv[2]
method = sys.argv[3]


def filter_signal(signal, threshold=1e3):
    fourier = rfft(signal)
    frequencies = rfftfreq(signal.size, d=20e-3/signal.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)

X = pd.DataFrame(X)
X_denoised = X.copy()

# train

# Apply filter_signal function to the data in each series
denoised_data = X.apply(lambda x: filter_signal(x))

df = denoised_data

df.columns = ['values']

X = df.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size-1:len(X)]

train_dates = currency.iloc[:len(train)].index
train_dates = pd.Series(train_dates.format())
train_dates = train_dates.str.replace('28', '01', regex=True)


test_dates = currency.iloc[len(train):].index
test_dates = pd.Series(test_dates.format())
test_dates = test_dates.str.replace('28', '01', regex=True)

prediction = pd.DataFrame()
dates = pd.concat([train_dates, test_dates]).reset_index(drop=True)
train_data = pd.concat([pd.DataFrame(train), pd.DataFrame(np.zeros((len(test)-1, 1)))]).reset_index(drop=True)
test_data = pd.concat([pd.DataFrame(np.zeros((len(train)-1, 1))), pd.DataFrame(test)]).reset_index(drop=True)
prediction = prediction.append(pd.DataFrame(dates, columns=['Date']))
prediction = pd.concat([prediction,train_data], axis=1, sort = False)
prediction = pd.concat([prediction,test_data], axis=1, sort = False)
prediction.columns = ['Date', 'Train', 'Test']

if "ARIMA" in stat:
    
    history = [x for x in train]
    predictions_arima_1 = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(0,1,1))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions_arima_1.append(yhat)
        obs = test[t]
        history.append(yhat)
    error = sqrt(mean_squared_error(test, predictions_arima_1))

    history = [x for x in train]
    predictions_arima_2 = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(0,1,1))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions_arima_2.append(yhat)
        obs = test[t]
        history.append(obs)
    error = sqrt(mean_squared_error(test, predictions_arima_2))
    
    arima_1 = pd.concat([pd.DataFrame(np.zeros((len(train)-1, 1))), pd.DataFrame(predictions_arima_1)]).reset_index(drop=True)
    arima_2 = pd.concat([pd.DataFrame(np.zeros((len(train)-1, 1))), pd.DataFrame(predictions_arima_2)]).reset_index(drop=True)  
       
    
    if 'Simple' in method:
            prediction = pd.concat([prediction,arima_1], axis=1, sort = False)
            columns = list(prediction.columns)
            prediction.columns = columns[:-1]+['ARIMA1']
            
    if 'Rolling' in method:
            prediction = pd.concat([prediction,arima_2], axis=1, sort = False)
            columns = list(prediction.columns)
            prediction.columns = columns[:-1]+['ARIMA2']
        
    
if "EXP" in stat:

    predictions_SES_1 = list()
    fit2 = SimpleExpSmoothing(np.asarray(train)).fit(smoothing_level=0.5,optimized=False)
    predictions_SES_1 = fit2.forecast(len(test))
    
    history = [x for x in train]
    predictions_SES_2 = list()
    for t in range(len(test)):
        fit2 = SimpleExpSmoothing(np.asarray(history)).fit(smoothing_level=0.5,optimized=False)
        output = fit2.forecast(len(test))
        yhat = output[0]
        predictions_SES_2.append(yhat)
        obs = test[t]
        history.append(obs)
    
    simp_1 = pd.concat([pd.DataFrame(np.zeros((len(train)-1, 1))), pd.DataFrame(predictions_SES_1)]).reset_index(drop=True)
    simp_2 = pd.concat([pd.DataFrame(np.zeros((len(train)-1, 1))), pd.DataFrame(predictions_SES_2)]).reset_index(drop=True)
    
    if 'Simple' in method:
            prediction = pd.concat([prediction,simp_1], axis=1, sort = False)
            columns = list(prediction.columns)
            prediction.columns = columns[:-1]+['EXP1']
            
    if 'Rolling' in method:
            prediction = pd.concat([prediction,simp_2], axis=1, sort = False)
            columns = list(prediction.columns)
            prediction.columns = columns[:-1]+['EXP2']


    
if "HOLT" in stat:

    
    predictions_WINTER_1 = list()
    fit1 = ExponentialSmoothing(np.asarray(train) ,seasonal_periods=4 ,trend='add', seasonal='add',).fit()
    predictions_WINTER_1 = fit1.forecast(len(test))
    
    
    
    history = [x for x in train]
    predictions_WINTER_2 = list()
    for t in range(len(test)):
        fit1 = ExponentialSmoothing(np.asarray(history) ,seasonal_periods=4 ,trend='add', seasonal='add',).fit()
        output = fit1.forecast(len(test))
        yhat = output[0]
        predictions_WINTER_2.append(yhat)
        obs = test[t]
        history.append(obs)

    holt_1 = pd.concat([pd.DataFrame(np.zeros((len(train)-1, 1))), pd.DataFrame(predictions_WINTER_1)]).reset_index(drop=True)
    holt_2 = pd.concat([pd.DataFrame(np.zeros((len(train)-1, 1))), pd.DataFrame(predictions_WINTER_2)]).reset_index(drop=True)

    if 'Simple' in method:
            prediction = pd.concat([prediction,holt_1], axis=1, sort = False)
            columns = list(prediction.columns)
            prediction.columns = columns[:-1]+['HOLT1']
            
    if 'Rolling' in method:
            prediction = pd.concat([prediction,holt_2], axis=1, sort = False)
            columns = list(prediction.columns)
            prediction.columns = columns[:-1]+['HOLT2']
            
            
prediction = prediction.set_index(['Date'])

#Insert data into cube      
cellset = {}
for date in prediction.index:
    for set_name in prediction.columns:
        if set_name == 'ARIMA1':
            cellset[('ARIMA', str(date), 'CHF', 'Simple')] = prediction.loc[date][set_name]
        
        if set_name == 'ARIMA2':
            cellset[('ARIMA', str(date), 'CHF', 'Rolling')] = prediction.loc[date][set_name]
        
        if set_name == 'EXP1':
            cellset[('EXP', str(date), 'CHF', 'Simple')] = prediction.loc[date][set_name]
        
        if set_name == 'EXP2':
            cellset[('EXP', str(date), 'CHF', 'Rolling')] = prediction.loc[date][set_name]
            
        if set_name == 'HOLT1':
            cellset[('HOLT', str(date), 'CHF', 'Simple')] = prediction.loc[date][set_name]
        
        if set_name == 'HOLT2':
            cellset[('HOLT', str(date), 'CHF', 'Rolling')] = prediction.loc[date][set_name]
        
        if set_name == 'Train':
            cellset[('ARIMA', str(date), 'CHF', 'Train')] = prediction.loc[date][set_name]
            cellset[('EXP', str(date), 'CHF', 'Train')] = prediction.loc[date][set_name]
            cellset[('HOLT', str(date), 'CHF', 'Train')] = prediction.loc[date][set_name]
          
        if set_name == 'Test':
            cellset[('ARIMA', str(date), 'CHF', 'Test')] = prediction.loc[date][set_name]
            cellset[('EXP', str(date), 'CHF', 'Test')] = prediction.loc[date][set_name]
            cellset[('HOLT', str(date), 'CHF', 'Test')] = prediction.loc[date][set_name]
          
cube = Cube('zSYS Analogic External Drivers Currency', ['zSYS Analogic External Drivers Currency Set', 'Currency Calendar Day', 'Currency','zSYS Analogic External Drivers Currency Measure'])
tm1.cubes.cells.write_values(cube.name, cellset)