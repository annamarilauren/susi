# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 13:35:00 2019

@author: alauren
"""
from susi_utils import read_FMI_weather
import datetime
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
mypath = r'C:/Users/alauren/Documents/WinPython-64bit-2.7.10.3/Susi_7_0_py27/wfilesmese/'

wfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

def growing_season(df, yr):
    """
    Process growing season weather data (YASSO-model needs this)
    Input: 
        weather data as pandas dataframe
        yr year as integer
    Output:
        growing season weather as dataframe
    """
    
    
    #---------------read weather data------------------------------
    start_date = str(yr) + '-05-01'                                                 # Time range starts from May 1st
    end_date = str(yr) + '-09-30'                                                   # And ends to Sept 30
    dftmp= df[str(yr)]     
    df_gs=dftmp[start_date:end_date].copy()
    return df_gs


start_date = datetime.datetime(1980,1,1)
end_date = datetime.datetime(1984,12,31)
T=[]; P=[]; gst =[]; gsp=[]
yrs = [1980, 1981, 1982, 1983, 1984]

for f in wfiles[:-1]:
    forc=read_FMI_weather(0, start_date, end_date, sourcefile=mypath+f)           # read weather input
    T.append(forc['T'].mean())
    P.append(forc['Prec'].sum()/5.)
    t = 0.; p=0.    
    for y in yrs:
        gs=growing_season(forc, y)
        t+= gs['T'].mean()/5.
        p += gs['Prec'].sum()/5.
        
    gst.append(t)
    gsp.append(p)
print np.mean(np.array(T))
print np.mean(np.array(P))
print np.mean(np.array(gst))
print np.mean(np.array(gsp))
