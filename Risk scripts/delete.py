import configparser
from datetime import timedelta, date
from TM1py.Objects import Cube, Dimension, Hierarchy, Element, ElementAttribute
from TM1py.Services import TM1Service
import numpy as np
from pandas import read_csv
from numpy.fft import *


config = configparser.ConfigParser()
config.read('D:/TM1 Models/AnalogicExternalDrivers/Python work/process_scripts/tm1conf.ini', encoding='utf-8')
tm1 = TM1Service(**config['Analogic External Drivers'])

cube = Cube('zSYS Analogic External Drivers Currency',
            ['zSYS Analogic External Drivers Currency Set',
             'Currency Calendar Year',
             'Currency',
             'zSyS Analogic External Drivers Currency Measure'])

if tm1.cubes.exists(cube.name):           
    tm1.cubes.delete(cube.name)

