from scipy.stats import bernoulli
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
from pert import PERT
import random
import matplotlib.pyplot as plt
import matplotlib
from TM1py import TM1Service
from TM1py import Utils
from TM1py.Objects import Cube, Dimension, Hierarchy, Element
from datetime import date, timedelta
import datetime
import configparser
from scipy import stats
import math
import sys


def bernoulli_freq(size, p):
    data_bern = bernoulli.rvs(size=size, p=p)
    return data_bern

def poisson_freq(mu, size):
    data_poisson = poisson.rvs(mu=mu, size=size)
    return data_poisson

def binomial_freq(mu, size):
    data_binomial = np.random.binomial(params[0],params[1], 1000)
    return data_binomial

def pert_sev(params):
    data_pert = PERT(params[0], params[1], params[2])
    return data_pert

def normal_sev(params):
    data_normal = np.random.normal(params[0], params[1], 1000)
    return data_normal

def triangle_sev(params):
    data_triangle = np.random.triangular(params[0],params[1],params[2], 1000)
    return data_triangle

def uniform_sev(params):
    data_uniform = np.random.uniform(params[0],params[1], 1000)
    return data_uniform

def exponential_sev(params):
    data_exponential = np.random.exponential(params[0], 1000)
    return data_exponential

config = configparser.ConfigParser()
tm1 = TM1Service(address='kseed-dc1.knowledgeseed.local', port=5960, user='admin', password='', ssl=False)

risk_name = sys.argv[1]
timeline = sys.argv[2]
lineitem = sys.argv[3]

StepNum = 1000
step_num = 1000


def run_simulation(risk_name, timeline, lineitem, step_num):

    view_name = 'Base + All'

    cube_name = 'Risk Event Info'
    mdx = 'SELECT {[Risk Event].['+risk_name+']} on ROWS, {[Measure Risk Event Info].[SelectedSeverity],[Measure Risk Event Info].[SelectedFrequency]} on COLUMNS  FROM [Risk Event Info] '
    dist = tm1.cubes.cells.execute_mdx_raw(mdx)
    values = dist.get('Cells')
    dist_severity = values[0].get('Value')
    dist_frequency = values[1].get('Value')
    
    cube_name = 'Risk Event SimParam'

    mdx = 'SELECT {[Simulation Config].[Frequency],[Simulation Config].[Severity]} on ROWS, {[Measure Risk Event SimParam].[Parameter1],[Measure Risk Event SimParam].[Parameter2],[Measure Risk Event SimParam].[Parameter3]} on COLUMNS  FROM [Risk Event Sim Param] WHERE ([Risk Event].['+risk_name+'],[Timeline].['+timeline+'],[Financial Statement Lineitem].['+lineitem+'])'
    param = tm1.cubes.cells.execute_mdx_raw(mdx)
    param_values = param.get('Cells')
    param_values
    freq_param = param_values[0].get('Value')
    sev_param = [param_values[3].get('Value'),param_values[4].get('Value'),param_values[5].get('Value')]
    
    if freq_param == None:
            freq_param = 0
    
    for i in range(0,3):
        if sev_param[i] == None:
            sev_param[i] = 0
            
    step_num = 1000
    size = step_num
    
    if dist_frequency == 'Bernoulli':
        p = freq_param
        p = p/100
        data_freq = bernoulli_freq(size, p)
        freq_hist_values, freq_hist_bins, freq_hist_bars = plt.hist(data_freq, bins = 10, weights=np.ones(len(data_freq)) / len(data_freq))
   
    
    if dist_frequency == 'Poisson':
        mu = freq_param
        data_freq = poisson_freq(mu, size)
        freq_hist_values, freq_hist_bins, freq_hist_bars = plt.hist(data_freq, bins = 10, weights=np.ones(len(data_freq)) / len(data_freq))

    if dist_severity == 'Pert':
        sev_param.sort()
        params = sev_param
        data_sev = pert_sev(params)
        data_sev = data_sev.rvs(1000)
        sev_hist_values, sev_hist_bins, sev_hist_bars = plt.hist(data_sev, bins=10, weights=np.ones(len(data_sev)) / len(data_sev))
        data_severity = data_sev
        data_to_plot = []
    
    if dist_severity == 'Normal':
        params = [sev_param[0], sev_param[1]]
        data_sev = pert_sev(params)
        data_sev = data_sev.rvs(1000)
        sev_hist_values, sev_hist_bins, sev_hist_bars = plt.hist(data_sev, bins=10, weights=np.ones(len(data_sev)) / len(data_sev))
        data_severity = data_sev
        data_to_plot = []
    
    if dist_severity == 'Triangle':
        sev_param.sort()
        params = sev_param
        data_sev = triangle_sev(params)
        sev_hist_values, sev_hist_bins, sev_hist_bars = plt.hist(data_sev, bins=10, weights=np.ones(len(data_sev)) / len(data_sev))
        data_severity = data_sev
        data_to_plot = []
        
    if dist_severity == 'Uniform':
        params = [sev_param[0], sev_param[1]]
        data_sev = uniform_sev(params)
        sev_hist_values, sev_hist_bins, sev_hist_bars = plt.hist(data_sev, bins=10, weights=np.ones(len(data_sev)) / len(data_sev))
        data_severity = data_sev
        data_to_plot = []
        
    if dist_severity == 'Exponential':
        params = [sev_param[0]]
        data_sev = exponential_sev(params)
        sev_hist_values, sev_hist_bins, sev_hist_bars = plt.hist(data_sev, bins=10, weights=np.ones(len(data_sev)) / len(data_sev))
        data_severity = data_sev
        data_to_plot = []
        
    comp_result = []

    for i in range(0, step_num):
        freq_value = random.choice(data_freq)
        
        if freq_value == 0:
            comp_result.append(0)
        else:
            comp_step = sum(random.sample(list(data_sev),freq_value))
            comp_result.append(comp_step)
            
    comp_hist_values, comp_hist_bins, comp_hist_bars = plt.hist(comp_result, bins=10, weights=np.ones(len(comp_result)) / len(comp_result))
    comp_result=np.array(comp_result)
    statistics = stats.describe(comp_result)
    
    percentiles = []
    percentiles_values = []
    
    for per in range(0,105,5):
        
        percentiles.append(per/100)
        percentiles_values.append(np.percentile(comp_result,per))
    
    plot_x = [0]
    plot_y = [0]
    x = plot_x + percentiles
    y = plot_y + percentiles_values
          

    '''Write Compound'''
    cellset = {}
    cube_name = 'Risk Simulation Result by Step'
    
    '''Write Frequency'''
    cellset = {}
    
    for i in range(1,len(data_freq)+1):
        cellset[(risk_name, timeline, lineitem, 'step'+str(i),'Frequency')] = data_freq[i-1]
    
    tm1.cubes.cells.write_values(cube_name, cellset)
    
    '''Write Severity'''
    
    for i in range(1,len(data_severity)+1):
        cellset[(risk_name, timeline, lineitem, 'step'+str(i),'Severity')] = data_severity[i-1]
    
    tm1.cubes.cells.write_values(cube_name, cellset)
    
    '''Write Cummul'''
    cellset = {}
    data_cummul = comp_result
    data_cummul.sort()
    for i in range(1,len(data_cummul)+1):
        cellset[(risk_name, timeline, lineitem, 'step'+str(i),'Cummul')] = data_cummul[i-1]
    
    tm1.cubes.cells.write_values(cube_name, cellset)
    
    
    '''Write Freq Hist'''
    cellset = {}
    for i in range(1,len(freq_hist_values)+1):
        cellset[(risk_name, timeline, lineitem, 'bin'+str(i),'Frequency')] = freq_hist_values[i-1]
    
    tm1.cubes.cells.write_values(cube_name, cellset)
                 
                 
    '''Write Freq Hist Left'''
    cellset = {}
    for i in range(1,len(freq_hist_bins)):
        cellset[(risk_name, timeline, lineitem, 'bin'+str(i),'BinLeft')] = freq_hist_bins[i-1]
    
    tm1.cubes.cells.write_values(cube_name, cellset)
    
    
    '''Write Comp Hist'''
    cellset = {}
    for i in range(1,len(comp_hist_values)+1):
        cellset[(risk_name, timeline, lineitem, 'bin'+str(i),'Compound')] = comp_hist_values[i-1]
    
    tm1.cubes.cells.write_values(cube_name, cellset)
    
    
    '''Write Comp Hist Left'''
    cellset = {}
    for i in range(1,len(comp_hist_bins)):
        cellset[(risk_name, timeline, lineitem, 'bin'+str(i),'CompoundBinLeft')] = comp_hist_bins[i-1]
    
    tm1.cubes.cells.write_values(cube_name, cellset)
    
                
    
    '''Write Sev Hist'''
    cellset = {}
    for i in range(1,len(sev_hist_values)+1):
        cellset[(risk_name, timeline, lineitem, 'bin'+str(i),'Severity')] = float(sev_hist_values[i-1])
    
    tm1.cubes.cells.write_values(cube_name, cellset)
    
    
    '''Write Sev Hist Left'''
    cellset = {}
    for i in range(1,len(sev_hist_bins)):
        cellset[(risk_name, timeline, lineitem, 'bin'+str(i),'SeverityBinLeft')] = sev_hist_bins[i-1]
    
    tm1.cubes.cells.write_values(cube_name, cellset)
    
    
    
    '''Write Stats'''
    
    step0_10 = sum(data_cummul[0:100])
    cellset = {}
    cube_name = 'Risk Simulation Result'
    cellset[(risk_name, timeline, lineitem,'Min')] = statistics.minmax[0]
    cellset[(risk_name, timeline, lineitem,'Max')] = statistics.minmax[1]
    cellset[(risk_name, timeline, lineitem,'Mean')] = statistics.mean
    cellset[(risk_name, timeline, lineitem,'STD Dev')] = math.sqrt(statistics.variance)
    cellset[(risk_name, timeline, lineitem,'Values')] = step_num
    cellset[(risk_name, timeline, lineitem,'MeanSensitivityMin')] = sum(data_cummul[0:100])/100
    cellset[(risk_name, timeline, lineitem,'MeanSensitivityMax')] = sum(data_cummul[900:1010])/100
    
    
    tm1.cubes.cells.write_values(cube_name, cellset)
       
        
run_simulation(risk_name, timeline, lineitem, step_num)