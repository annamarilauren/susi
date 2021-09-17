# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:10:42 2020

@author: alauren
"""
import numpy as np
import pandas as pd
import datetime
from susi_utils import  get_motti, read_FMI_weather, get_mese_input
from susi_para import get_susi_para
from susi83 import run_susi
import susi_io

#def call_local_susi():
    #from dwts_para import para
"""
single run, all input here
switches in susi83: Opt_strip= True, no output, only figs
"""    
#***************** local call for SUSI*****************************************************
folderName=r'C:/Users/alauren/Documents/WinPython-64bit-2.7.10.3/Susi_8_3_py37/outputs/' #'sensitivity/'
susiPath = r'C:/Users/alauren/Documents/Susi_9/'
wpath = r'C:/Users/alauren/Documents/Susi_9/'
mottipath =  r'C:/Users/alauren/Documents/Susi_9/'

#mf='motti viitasaari_mtkg.xls'
mf='susi_stand_ccf.xlsx'

wdata='parkano_weather.csv'

start_date = datetime.datetime(2005,1,1)
end_date=datetime.datetime(2007,12,31)
start_yr = start_date.year 
end_yr = end_date.year
yrs = (end_date - start_date).days/365.25
mottifile = mottipath + mf
df = get_motti(mottifile)
sfc =  3                                                                         #soil fertility class
ageSim = 50.         
sarkaSim = 40. 
n = int(sarkaSim / 2)
        
site = 'develop_scens'

forc=read_FMI_weather(0, start_date, end_date, sourcefile=wpath+wdata)           # read weather input
            
wpara, cpara, org_para, spara, outpara, photopara = get_susi_para(wlocation='undefined', peat=site, 
                                                                          folderName=folderName, hdomSim=None,  
                                                                          ageSim=ageSim, sarkaSim=sarkaSim, sfc=sfc, 
                                                                          susiPath=susiPath,
                                                                          n=n)
                                                                          
    
v_ini, v, iv5,  cbt, dcbt, cb, dcb, w,dw,logs,pulp, dv,dlogs,dpulp,yrs, bmgr,  \
                                Nleach, Pleach, Kleach, DOCleach, runoff, \
                                nrelease, prelease,krelease, ch4release = run_susi(forc, wpara, cpara, 
                                org_para, spara, outpara, photopara, start_yr, end_yr, wlocation = 'undefined', 
                                mottifile=mottifile, peat= 'other', photosite='All data', 
                                folderName=folderName,ageSim=ageSim, sarkaSim=sarkaSim, sfc=sfc, susiPath=susiPath)
    
          
             
#call_local_susi()

