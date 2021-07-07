# -*- coding: utf-8 -*-
"""
Created on Tue Aug 08 10:38:45 2017

@author: lauren
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pylab as plt
import pickle
from scipy.misc import derivative
from scipy.interpolate import InterpolatedUnivariateSpline as interS
from pyproj import Proj, transform

class SetupException (Exception):
    pass

def peat_hydrol_properties(x, unit='g/cm3', var='bd', ptype='A'):
    """
    Peat water retention and saturated hydraulic conductivity as a function of bulk density
    Päivänen 1973. Hydraulic conductivity and water retention in peat soils. Acta forestalia fennica 129.
    see bulk density: page 48, fig 19; degree of humification: page 51 fig 21
    Hydraulic conductivity (cm/s) as a function of bulk density(g/cm3), page 18, as a function of degree of humification see page 51 
    input:
        - x peat inputvariable in: db, bulk density or dgree of humification (von Post)  as array \n
        - bulk density unit 'g/cm3' or 'kg/m3' \n
        - var 'db' if input variable is as bulk density, 'H' if as degree of humification (von Post) \n
        - ptype peat type: 'A': all, 'S': sphagnum, 'C': Carex, 'L': wood, list with length of x 
    output: (ThetaS and ThetaR in m3 m-3)
        van Genuchten water retention parameters as array [ThetaS, ThetaR, alpha, n] \n
        hydraulic conductivity (m/s)
    """
    #paras is dict variable, parameter estimates are stored in tuples, the model is water content = a0 + a1x + a2x2, where x is
    para={}                                                                     #'bd':bulk density in g/ cm3; 'H': von Post degree of humification
    para['bd'] ={'pF0':(97.95, -79.72, 0.0), 'pF1.5':(20.83, 759.69, -2484.3),
            'pF2': (3.81, 705.13, -2036.2), 'pF3':(9.37, 241.69, -364.6),
            'pF4':(-0.06, 249.8, -519.9), 'pF4.2':(0.0, 174.48, -348.9)}
    para['H'] ={'pF0':(95.17, -1.26, 0.0), 'pF1.5':(46.20, 8.32, -0.54),
            'pF2': (27.03, 8.14, -0.43), 'pF3':(17.59, 3.22, -0.07),
            'pF4':(8.81, 3.03, -0.10), 'pF4.2':(5.8, 2.27, -0.08)}
    
    intp_pF1={}                                                                 # interpolation functions for pF1        
    intp_pF1['bd'] = interp1d([0.04,0.08,0.1,0.2],[63.,84.,86.,80.],fill_value='extrapolate')
    intp_pF1['H'] = interp1d([1.,4.,6.,10.],[75.,84.,86.,80.],fill_value='extrapolate')
    
    #Saturatated hydraulic conductivity parameters
    Kpara ={'bd':{'A':(-2.271, -9.80), 'S':(-2.321, -13.22), 'C':(-1.921, -10.702), 'L':(-1.921, -10.702)}, 
            'H':{'A':(-2.261, -0.205), 'S':(-2.471, -0.253), 'C':(-1.850, -0.278), 'L':(-2.399, -0.124)}}
    
    vg_ini=(0.88,	0.09, 0.03, 1.3)                                              # initial van Genuchten parameters (porosity, residual water content, alfa, n)

    x = np.array(x)
    prs = para[var]; pF1=intp_pF1[var]
    if unit=='kg/m3'and var=='db': x=x/1000.
    if  np.shape(x)[0] >1 and len(ptype)==1:
        ptype=np.repeat(ptype, np.shape(x)[0])        
    vgen = np.zeros((np.size(x),4))
    Ksat = np.zeros((np.size(x)))
    
    #wcont = lambda x, (a0, a1, a2): a0 + a1*x + a2*x**2.
    wcont = lambda x, *a: a[0] + a[1]*x + a[2]*x**2.
    van_g = lambda pot, *p:   p[1] + (p[0] - p[1]) / (1. + (p[2] * pot) **p[3]) **(1. - 1. / p[3])   
    #K = lambda x, (a0, a1): 10.**(a0 + a1*x) / 100.   # to m/s   
    K = lambda x, *a: 10.**(a[0] + a[1]*x) / 100.   # to m/s   
    
    potentials =np.array([0.01, 10.,32., 100.,1000.,10000.,15000. ])
    
    wc = (np.array([wcont(x,*prs['pF0']), pF1(x), wcont(x,*prs['pF1.5']), wcont(x,*prs['pF2']),
               wcont(x,*prs['pF3']), wcont(x,*prs['pF4']),wcont(x,*prs['pF4.2'])]))/100.
        
    for i,s in enumerate(np.transpose(wc)):
        vgen[i],_= curve_fit(van_g,potentials,s, p0=vg_ini)                      # van Genuchten parameters
        
    for i, a, pt in zip(range(len(x)), x, ptype):
        Ksat[i] = K(a, *Kpara[var][pt])                                          # hydraulic conductivity (cm/s -> m/s) 
    
    return vgen, Ksat

def CWTr(nLyrs, z, dz, pF, Ksat, direction='positive'):
    """
    Returns interpolation functions 
        sto=f(gwl)  profile water storage as a function ofground water level
        gwl=f(sto)  ground water level
        tra=f(gwl)  transissivity
    Input:
        nLyrs number of soil layers
        d depth of layer midpoint
        dz layer thickness
        pF van Genuchten water retention parameters: ThetaS, ThetaR, alfa, n
        Ksat saturated hydraulic conductivity in m s-1
        direction: positive or negative downwards
    """    
    #-------Parameters ---------------------
    z = np.array(z)   
    dz =np.array(dz)
    nroot = 8   # 8 good number of layers in rooting zone
    #nroot =10      #for jääli simulations
    nroot2 = 3  #2 10 cm root layers for air-filled porosity

    #--------- Connection between gwl and water storage------------
    d = 6 if direction == 'positive' else -6   
    gwl=np.linspace(0,d,150)
    if direction == 'positive':
        sto = [sum(wrc(pF, x = np.minimum(z-g, 0.0))*dz) for g in gwl]     #equilibrium head m
        storoot = [np.sum(wrc(pF, x = np.minimum(z-g, 0.0))[0:nroot]*dz[0:nroot]) for g in gwl]
        storoot2 = [np.sum(wrc(pF, x = np.minimum(z-g, 0.0))[0:nroot2]*dz[0:nroot2]) for g in gwl]
    else:
        sto = [sum(wrc(pF, x = np.minimum(z+g, 0.0))*dz) for g in gwl]     #equilibrium head m
        storoot = [np.sum(wrc(pF, x = np.minimum(z+g, 0.0))[0:nroot]*dz[0:nroot]) for g in gwl]
        storoot2 = [np.sum(wrc(pF, x = np.minimum(z+g, 0.0))[0:nroot2]*dz[0:nroot2]) for g in gwl]

    gwlToSto = interp1d(np.array(gwl), np.array(sto), fill_value='extrapolate')
    airtot = sto[0]-sto                                                         #m air in the profile
    airroot = storoot[0]-storoot                                                #m air in the rooting zone
    afproot = (storoot2[0]-storoot2)/(sum(dz[:nroot2]))                         #air-filled porosity in root layer
    ratio = airroot[1:]/airtot[1:]                                            #share of air-filled porosity in rooting zone to total air volume
    sto = list(sto); gwl= list(gwl); ratio=list(ratio); afproot = list(afproot)         
    sto.reverse(); gwl.reverse(); ratio.reverse(); afproot.reverse()
    stoToGwl =interp1d(np.array(sto), np.array(gwl), fill_value='extrapolate')
    gwlToRatio = interp1d(np.array(gwl[1:]), np.array(ratio), fill_value='extrapolate' )
    gwlToAfp= interp1d(np.array(gwl), np.array(afproot), fill_value='extrapolate' )
    C = interp1d(np.array(gwl), np.array(np.gradient(gwlToSto(gwl))/np.gradient(gwl)), fill_value='extrapolate')  #storage coefficient function      
    
    del gwl, sto, ratio, afproot
        
    #----------Transmissivity-------------------
    K=np.array(Ksat*86400.)   #from m/s to m/day
    tr =[sum(K[t:]*dz[t:]) for t in range(nLyrs)]        
    if direction=='positive':        
        gwlToTra = interS(z, np.array(tr))            
    else:
        z= list(z);  z.reverse(); tr.reverse()
        gwlToTra = interS(-np.array(z), np.array(tr))                    
    del tr
    return gwlToSto, stoToGwl, gwlToTra, C, gwlToRatio, gwlToAfp

def wrc(pF, x=None, var=None):
    """
    vanGenuchten-Mualem soil water retention curve\n
    IN:
        pF - dict['ThetaS': ,'ThetaR': ,'alpha':, 'n':,] OR
           - list [ThetaS, ThetaR, alpha, n]
        x  - soil water tension [m H2O = 0.1 kPa]
           - volumetric water content [vol/vol]
        var-'Th' is x=vol. wat. cont.
    OUT:
        res - Theta(Psii) or Psii(Theta)
    NOTE:\n
        sole input 'pF' draws water retention curve and returns 'None'. For drawing give only one pF-parameter set. 
        if several pF-curves are given, x can be scalar or len(x)=len(pF). In former case var is pF(x), in latter var[i]=pf[i,x[i]]
               
    Samuli Launiainen, Luke 2/2016
    """
    if type(pF) is dict: #dict input
        #Ts, Tr, alfa, n =pF['ThetaS'], pF['ThetaR'], pF['alpha'], pF['n']
        Ts=np.array(pF['ThetaS'].values()); Tr=np.array( pF['ThetaR'].values()); alfa=np.array( pF['alpha'].values()); n=np.array( pF['n'].values())
        m= 1.0 -np.divide(1.0,n)
    elif type(pF) is list: #list input
        pF=np.array(pF, ndmin=1) #ndmin=1 needed for indexing to work for 0-dim arrays
        Ts=pF[0]; Tr=pF[1]; alfa=pF[2]; n=pF[3] 
        m=1.0 - np.divide(1.0,n)
    elif type(pF) is np.ndarray:
        Ts, Tr, alfa, n = pF.T[0], pF.T[1], pF.T[2], pF.T[3]
        m=1.0 - np.divide(1.0,n)
    else:
        print ('Unknown type in pF')
        
    def theta_psi(x): #'Theta-->Psi'
        x=np.minimum(x,Ts) 
        x=np.maximum(x,Tr) #checks limits
        s= ((Ts - Tr) / (x - Tr))#**(1/m)
        Psi=-1e-2/ alfa*(s**(1/m)-1)**(1/n) # in m
        return Psi
        
    def psi_theta(x): # 'Psi-->Theta'
        x=100*np.minimum(x,0) #cm
        Th = Tr + (Ts-Tr)/(1+abs(alfa*x)**n)**m
        return Th           
 
    if var is 'Th': y=theta_psi(x) #'Theta-->Psi'           
    else: y=psi_theta(x) # 'Psi-->Theta'          
    return y

        
def z_distrib_decomposition(gwl, Qdistrib_beta = 0.93): # Alkuperäinen 0.88, Koukkujoki 0.87
    """
    Distribution Gale & Grigal Canadian Journal of Forest Research, 1987, 17(8): 829-834, 10.1139/x87-131  \n
    Input:
        gwl m
   Output:
        share of actual decomposition from the potential
    """
    return  Qdistrib_beta**0.0 - Qdistrib_beta**(np.abs(gwl*100.))

def potential_peat_heterotrophic_respiration(T, sfc, V):
    """
    Returns potential soil oxygen consumption through soil respiration (CO2) flux through the soil surface according to Ojanen et al 2010 ForEco, eq1
    Boundary conditions: max soil temperature 16 deg C. No minimum value. \n
    Input:
        Tsoil soil temperature in 5 cm depth, deg C \n
        spara site and soil parameters 
    Output:
        Potential O2 consumption flux to soil as kg m-2 s-1, refers to level demanded by total respiration \n
        Total respiration: Potential CO2 efflux from soil as kg m-2 s-1 \n
        Heterotrophic respiration, CO2 efflux caused by decomposition: Potential CO2 efflux from soil as kg m-2 s-1 \n
    """
    #sfc=spara['sfc']
    
    #Ojanen et al. 2010 Forest Ecology and Management 260:411-421            
    if T > 16.0: T=16.0
    #Computes momentary Total CO2 flux as a function of soil temperature and peat bulk density
    B= 350.0; T5ref=10.0; T50=-46.02
    #   Huom Ojanen 2010 B=350. Table 2    
    #Adjusted with bulk density
    wtm =80.0                                                                   # wtm mean water table in cm May-Oct, maximum 80 in the Ojanen data, max used becase later cut with the current wt
    pt = 99.0 
    #-----properties according to soil fertility class sfc:                                                                  #peat depth cm
    bd = {2: 0.14, 3: 0.11, 4: 0.10, 5: 0.08}           # Mese study: bulk densities in different fertility classes                                                                 # peat layer thickness, cm            

    wtm = 50.0                                                          #Huikari & Paarlahti 1968 Comm Inst For Fenn 64.1, Max growth at DWT 50 cm  
    Rref = 0.28 + V*5.9*10**-4 + bd[sfc]*1000.0 * 9.1*10**-4 +wtm * 1.5*10**-3  #Parameters: Table 3 RTot
    TotCO2 = Rref*np.exp(B*(1.0/(T5ref-T50)-1.0/(T-T50)))               #g m-2 h-1       
    TotCO2 = TotCO2 / 1000.0 / 3600.0                                   #Conversion to kg m-2 s-1
    O2 = TotCO2 / 1.375                                                 #Conversion from CO2 to O2 with mole mass ratio

    #Computes momentary Heterotrophic CO2 flux as a function of soil temperature and peat bulk density
    Rref = 0.0695 + V*3.7*10**-4 + bd[sfc]*1000.0 * 5.4*10**-4 +wtm * 1.2*10**-3  #Parameters: Table 3 RHet
    TASmr = 10.4                                                        #mean temperature may-october, deg C
    pt = 99.0                                                           # peat layer thickness, cm            
    B = 156.032 + 16.5*TASmr - 0.196*pt + 0.354*bd[sfc]                      #Ojanen 2010 et al. Table 4             
    HetCO2 = Rref*np.exp(B*(1.0/(T5ref-T50)-1.0/(T-T50)))               #g m-2 h-1       
    HetCO2 = HetCO2 / 1000.0 / 3600.0                                   #Conversion to kg m-2 s-1            
                                                                        #return: final converstaion into kg/ha/day
    return HetCO2*86400.*10000.

def read_FMI_weather(ID, start_date, end_date, sourcefile=None):
    """ 
    reads FMI interpolated daily weather data from file 
    IN: 
        ID - sve catchment ID. set ID=0 if all data wanted
        start_date - 'yyyy-mm-dd'
        end_date - 'yyyy-mm-dd'
    OUT:
        fmi - pd.dataframe with datetimeindex
            fmi columns:['ID','Kunta','aika','lon','lat','T','Tmax','Tmin','Prec','Rg',
                         'h2o','dds','Prec_a','Par','RH','esa','VPD','doy']
            units: T, Tmin, Tmax, dds[degC], VPD, h2o,esa[kPa], Prec, Prec_a[mm], Rg,
                   Par[Wm-2],lon,lat[deg]
    """
    
    #OmaTunniste;OmaItä;OmaPohjoinen;Kunta;siteid;vuosi;kk;paiva;longitude;latitude;t_mean;t_max;t_min;
    #rainfall;radiation;hpa;lamposumma_v;rainfall_v;lamposumma;lamposumma_cum
    #-site number
    #-date (yyyy mm dd)
    #-latitude (in KKJ coordinates, metres)
    #-longitude (in KKJ coordinates, metres)
    #-T_mean (degrees celcius)
    #-T_max (degrees celcius)
    #-T_min (degrees celcius)
    #-rainfall (mm)
    #-global radiation (per day in kJ/m2)
    #-H2O partial pressure (hPa)

    #ID=0
    print (' + Reading meteorological input file from ')
    print ('    -', sourcefile)
    #ID=0

    ## removed 'OmaTunniste', not sure if it is needed
    fi_colnames = ['Kunta','aika','longitude','latitude',
                   't_mean','t_max','t_min','rainfall','radiation','hpa']

    ## removed 'id', not sure if it is needed
    en_colnames = ['site','time','longitude','latitude',
                   't_mean','t_max','t_min','rainfall','radiation','hpa']

    first_line = None
    with open(sourcefile) as f:
        first_line = f.readline()

    ## note: original reader specified: encoding= 'ISO-8859-1'
    ## in case you still use such antiquated encoding, you may need to add it to the
    ## pandas.read_csv()s, preferably with encoding detection via chardet before that.
    if all([k in first_line for k in fi_colnames]):
        fmi = pd.read_csv(sourcefile, sep=None, usecols=fi_colnames, engine='python')
        time = pd.to_datetime(fmi['aika'],format='%Y%m%d')
        fmi.index = time
        fmi = fmi.rename(columns={'aika': 'time', 'Kunta': 'site'})
    elif all([k in first_line for k in en_colnames]):
        fmi = pd.read_csv(sourcefile, sep=None, usecols=en_colnames, engine='python')
        time = pd.to_datetime(fmi['time'],format='%Y%m%d')
        fmi.index = time
    else:
        raise SetupException("weather file " + sourcefile +
                             " is missing excepted column names")

    ## removed 'OmaTunniste': 'ID',
    fmi = fmi.rename(columns={'longitude':'lon',
                              'latitude':'lat',
                              't_mean':'T',
                              't_max':'Tmax',
                              't_min':'Tmin',
                              'rainfall':'Prec',
                              'radiation':'Rg',
                              'hpa':'h2o'})

    ## checking that radiation values are strictly positive
    ## TODO: should in fact check validity of all input data
    if((fmi['Rg'].values <= 0.0).any()):
        print("warning: replacing zero radiation values with 1.0")
        fmi[fmi['Rg'] <= 0.0] = 1.0

    fmi['h2o'] = 1.0e-1 * fmi['h2o']      # hPa-->kPa
    fmi['Rg'] = 1.0e3/86400.0 * fmi['Rg'] # kJ/m2/d-1 to Wm-2 
    fmi['Par'] = 0.5 * fmi['Rg']

    ## saturated vapor pressure    
    esa        = 0.6112*np.exp((17.67*fmi['T'])/(fmi['T'] + 273.16 - 29.66))  #kPa
    vpd        = esa - fmi['h2o']; #kPa   
    vpd[vpd<0] = 1.0e-5
    rh         = 100.0*fmi['h2o']/esa;
    rh[rh<0]   = 1e-6; rh[rh>100]=100.0
                
    fmi['RH']  = rh;
    fmi['esa'] = esa;
    fmi['vpd'] = vpd
    fmi['doy'] = fmi.index.dayofyear
    fmi        = fmi.drop(['time'],1)

    #replace nan's in prec with 0.0
    fmi['Prec'].fillna(value=0.0)    
    #del dat, fields, n, k, time
    
    ## get desired period
    fmi = fmi[(fmi.index >= start_date) & (fmi.index <= end_date)]
    ## note: don't know why this is needed, removing for now
    ##if ID > 0:
    ##    fmi=fmi[fmi['ID']==ID]

    return fmi
    
def nutrient_release(sfc, co2release, N = None, P = None, K = None):
    #sfc=spara['sfc']                                                            # soil fertility class    
    Nd =  {2: 1.9, 3: 1.6, 4: 1.4, 5: 1.2}                                       # Mese study: N cont in OM % dm
    Pd =  {2: 0.1, 3: 0.08, 4: 0.06, 5: 0.05}                                    # Mese study: P cont in OM % dm
    Kd =  {2: 0.041, 3: 0.04, 4: 0.035, 5: 0.03}                                 # Mese study: P cont in OM % dm (alkuperäinen)
            
    #Ndeposition = 5.                            #2-6 kg/ha/yr    Palviainen & Finer 2012 
    #Pdeposition = 0.12                         # 0.04-0.25
    #Kdeposition = 0.5                           # 0.3-0.6 kg/ha/yr    
    
    if N is None: N = Nd[sfc]
    if P is None: P = Pd[sfc]
    if K is None: K = Kd[sfc]

    C_in_OM = 0.55                                                              # C content in OM kg kg-1
    CO2_to_C = 12./44.
    Nmicrob = 0.8                                                               # microbial immobilisation    
    Pmicrob = 0.7
    Kmicrob = 0.0                                                               # microbial immobilisation    
    
    Nt = co2release * CO2_to_C / C_in_OM * N / 100. * (1.-Nmicrob) #+ Ndeposition   # Nrelease kg ha-1 day-1
    Pt = co2release * CO2_to_C / C_in_OM * P / 100. * (1.-Pmicrob) #+  Pdeposition # Prelease kg ha-1 day-1
    Kt = co2release * CO2_to_C / C_in_OM * K / 100. * (1.-Kmicrob)  #+ Kdeposition  # Prelease kg ha-1 day-1

    return Nt, Pt, Kt

def diff_nutrient_release(co2release, vol, spara, wpara, outpara, dates, wlocation, \
                summer_dwt, co2_respi, ditch_depth,rounds, yi, bm, npps, bmToVol, NtoVol, PtoVol, KtoVol):
           
    import susi_io
    """
    Computes nutrient release in each drain depth scenario, calculates the difference between the scenarios, and extends \n
    the difference into growth responses.
    Input:
        - co2release: daily mean value (accross the nodes in strip) kg/ha/day
        - vol: stand volume m3/ha daily basis
        - spara soil parameters
        - scen scenario name
    """
    gr_response=[]
    sfc=spara['sfc']; sp = spara['species']    
    N =  {2: 1.9, 3: 1.6, 4: 1.4, 5: 1.2}                                       # Mese study: N cont in OM % dm
    P =  {2: 0.1, 3: 0.08, 4: 0.06, 5: 0.05}                                    # Mese study: P cont in OM % dm
    K =  {2: 0.041, 3: 0.04, 4: 0.035, 5: 0.03}                                 # Mese study: P cont in OM % dm (alkuperäinen)
    C_in_OM = 0.55                                                              # C content in OM kg kg-1
    CO2_to_C = 12./44.
    Nmicrob = 0.8                                                               # microbial immobilisation    
    Pmicrob = 0.7
    Kmicrob = 0.0                                                               # microbial immobilisation    
    
    print ('¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨')
    for c in range(np.shape(co2release)[0]):
        co2=co2release[c,:]
        Nt = co2 * CO2_to_C / C_in_OM * N[sfc] / 100. * (1.-Nmicrob)   # Nrelease kg ha-1 day-1
        Pt = co2 * CO2_to_C / C_in_OM * P[sfc] / 100. * (1.-Pmicrob)   # Prelease kg ha-1 day-1
        Kt = co2 * CO2_to_C / C_in_OM * K[sfc] / 100. * (1.-Kmicrob)   # Prelease kg ha-1 day-1
        print ('Released nutrients N', np.round(sum(Nt),2),'P', np.round(sum(Pt), 2),'K', np.round(sum(Kt),2))
        gN = NtoVol(sum(Nt))
        gP = PtoVol(sum(Pt))
        gK = KtoVol(sum(Kt))
        print ('Growths', np.round(gN,2), np.round(gP,2), np.round(gK,2))
    
    print ('¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨')    
    
    print ('***************************************')
    print ('Comparing scenarios:')    
    comparisons = np.shape(co2release)[0]
    vol_end=vol[0]+(yi[-1]-yi[0])
    for c in range(comparisons)[1:]:        
        diff = co2release[c]-co2release[0]
        Nrel = diff * CO2_to_C / C_in_OM * N[sfc] / 100. * (1.-Nmicrob)   # Nrelease kg ha-1 day-1
        Prel = diff * CO2_to_C / C_in_OM * P[sfc] / 100. * (1.-Pmicrob)   # Prelease kg ha-1 day-1
        Krel = diff * CO2_to_C / C_in_OM * K[sfc] / 100. * (1.-Kmicrob)   # Prelease kg ha-1 day-1

        yrs =(dates[-1]- dates[0]).days/365.

        tg =  vol[-1]-vol[0]
        ag = bmToVol(bm[0]+sum(npps[c,:])) -bmToVol(bm[0])
        print ('  +Control vs',spara['scenario name'][c])
        print ('    -Nutrient release:','N', np.round(sum(Nrel),3),'P', np.round(sum(Prel),3),'K', np.round(sum(Krel),3), 'kg/ha')
        print ('    -Volume control', np.round(vol[-1],1), 'increment: min (',np.round(gN,2), np.round(gP,2), np.round(gK,2), ')')             
        print ('    -Increment per yr: min(', np.round(gN/yrs,1), np.round(gP/yrs,1), np.round(gK/yrs,1), ')')        
        print ('    -Years in simulation', np.round(yrs))
        print ('    -Table growth', np.round(tg,2), 'm3/ha')
        print ('    -Assimilation growth', np.round(ag,2), 'm3/ha')
        print ('    -Difference', np.round(ag-tg,2), 'm3/ha')
        print ('    -Table max growth', np.round(tg*0.15), 'm3/ha')
        
        limiting = min(gN, gP, gK)
        gr_response.append(limiting)
        
        if c==range(comparisons)[-1]:
           gr_response = np.ravel(gr_response)    
           summer_dwt=np.array(summer_dwt); co2_respi=np.array(co2_respi); 
           gr_response=np.array(gr_response); ditch_depth=np.array(ditch_depth)    
           gr_response= np.insert(gr_response, 0,0.0)
           fwt = interp1d(summer_dwt, ditch_depth, bounds_error=False, fill_value='extrapolate')
           fgr = interp1d(ditch_depth, gr_response, bounds_error=False, fill_value='extrapolate')
           cr_depth = fwt(0.35); gr_crd = fgr(fwt(0.35))
        else:    
            cr_depth = 0; gr_crd=0
        susi_io.write_gr_excel(wlocation, wpara, spara, outpara, gN, gP, gK, c,cr_depth, gr_crd)
        
        po=False
        if po:
            fig= plt.figure(num = 'Control vs' + spara['scenario name'][c], \
                facecolor=(232/255.0, 243/255.0, 245.0/255), edgecolor='k',figsize=(18.0,12.0))   #Figsize(w,h), tuple inches 
            ax = fig.add_axes([0.05, 0.5, 0.55, 0.46]) #left, bottom, width, height
            plt.ylabel('stand volume')        
            plt.plot(dates, vol, 'k-')
            plt.plot(dates, vol+np.cumsum(diff)/sum(diff)*limiting, 'r-')
            plt.fill_between(dates, vol, vol+np.cumsum(Krel*gK), color='red', alpha=0.5)
        
    return gr_response

def critical_ditch_depth(summer_dwt, ditch_depth, growth_response):
    fwt = interp1d(summer_dwt, ditch_depth, bounds_error=False, fill_value='extrapolate')
    fgr = interp1d(ditch_depth, growth_response, bounds_error=False, fill_value='extrapolate')
    return fwt(-0.35), fgr(fwt(-0.35))

def get_mese_input(ifile):
        
    df = pd.read_excel(ifile, sheet_name=0, usecols=range(28), skiprows=1, header=None )
    cnames = ['nro','koe',	'kaista','ddist','age_ini','stripw','dd_west','dd_east','sfc',	'bd','n_mg/g','p_mg/g','k_mg/g','V_tot5',	
              'wfile',	'Vtot_0','Ntot_0','H_0	','iv_tot5','wtmax','wtmed','wtmin', 'Ndem', 'Pdem', 'Kdem', 'depoN', 'depoP', 'depoK']
    df = pd.read_excel(ifile, sheet_name=0, usecols=range(28), skiprows=1, header=None )
    df.columns=cnames
    df = pd.DataFrame(df).set_index('nro')

    return df

def get_mese_out(ifile):
    df = pd.read_excel(ifile, sheet_name=0, usecols=range(24), skiprows=1, header=None )
    cnames = ['nro','v_ini', 'v','iv5',	'Nsup', 'Psup','Ksup','Crel','dwt_loc','cb', 'cbt','sfc']
    df = pd.read_excel(ifile, sheet_name=0, usecols=range(10), skiprows=1, header=None )
    df.columns=cnames
    df = pd.DataFrame(df).set_index('nro')
    return df

def get_mese_scen(ifile):
    #df = pd.read_excel(ifile, sheetname=0, parse_cols=range(26), skiprows=1, header=None )
    #cnames = ['nro', 'v_ini', 'v_end_30', 'v_end_50','v_end_70', 'v_end_90', 'dg_30', 'dg_50', 'dg_70', 'dg_90', 'cb_30',	
    #          'cb_50', 'cb_70', 'cb_90', 'dcb_30', 'dcb_50', 'dcb_70', 'dcb_90', 'wt_30', 'wt_50', 'wt_70', 'wt_90',	
    #          'dwt_30', 'dwt_50', 'dwt_70', 'dwt_90', 'sfc', 'koe', 'dead_w','cbn']
    #df = pd.read_excel(ifile, sheetname=0, parse_cols=range(30), skiprows=1, header=None )
    #df.columns=cnames

    df = pd.read_excel(ifile, sheet_name=0)
    
    df = pd.DataFrame(df).set_index('nro')
    return df


def get_motti(ifile, return_species_number = False):
    """read the Motti-simulation to be used as a basis for the Susi-simulation"""

    csv_extensions = [".csv",".CSV"]
    xls_extensions = [".xls",".XLS","xlsx","XLSX"]

    cnames = ['yr', 'age', 'N', 'BA', 'Hg', 'Dg', 'hdom', 'vol', 'logs',
              'pulp', 'loss', 'yield','mortality', 'stempulp', 'stemloss',
              'branch_living', 'branch_dead', 'leaves', 'stump', 'roots_coarse',
              'roots_fine']

    species_number = -1 ## return -1 if not possible to infer
    if ifile[-4:] in csv_extensions:
        df = pd.read_csv(ifile)
        if not set(cnames).issubset(df.columns):
            raise SetupException("stand file " + ifile +
                                 " does not have the expected columns")
    elif ifile[-4:] in xls_extensions:
        df = pd.read_excel(ifile, sheet_name=0, usecols=range(22), skiprows=1, header=None)
        df = df.drop([0], axis=1)
        df.columns = cnames
        try:
            cname=['idSpe']
            df2 = pd.read_excel(ifile, sheet_name=1, usecols=[4], skiprows=1, header=None)
            df2.columns = cname
        except:
            pass
        else:
            species_number = df2['idSpe'][0]
    else:
        raise SetupException("the stand file extension should be one " +
                             "of (.csv, .CSV, .xls, .XLS)")

    df = df[df['age']!=0] # remove rows with zero age

    ## note: not sure when the following would apply, disabling for now
    ## find thinnings and add a small time to lines with the age to enable interpolation
    ##steps = np.array(np.diff(df['age']), dtype=float)
    ##idx = np.ravel(np.argwhere(steps<1.))+1
    ##df['age'][idx]=df['age'][idx]+5./365.
    
    if return_species_number:
        return df, species_number
    else:
        return df

def nut_to_vol(vol_ini, Nrel,Prel,Krel,litter_mass, gvNup, gvPup, gvKup, leafmass, gv_leafmass, species='Pine'):
    """
    Input:
        vol_ini -  initial volume m3 ha-1
        Nrel, Prel, Krel - available mass of N, P, K (release from decomp + deposition) kg ha-1
        litter_mass - litter mass within the time step. kg ha-1
        gvNup, gvPup, gvKup - uptake of N P and K by ground vegetation kg ha-1 in time step
        leafmass, gv_leafmass  - tree leafmass, ground vegetation leafmass kg ha-1
        species - main tree species: pine, spruce, birch (other deciduous)
    Output
        volN, volP, volK - new volumes allowed by nutrient release 
    """
    gvshare = np.mean(gv_leafmass / (leafmass + gv_leafmass))     # maximum share of ground vegetation from the nutrient uptake -> shared in proportion of green biomass
    littershare = (1.0-gvshare)/2.                                # maximum share of nutrient supply allocated for litter
    
    MarjoNut = lambda vol, lna, b, k: np.exp(lna + b*np.log(vol) + k)
    MarjoBck = lambda nut, lna, b, k: np.exp((np.log(nut)-lna-k)/b)

    #'N':{'pine':[1.856,0.631,0.050]},
    #'P':{'pine':[-2.387,0.754,0.158]},
    #'K':{'pine':[0.839,0.650,0.070]}        

    par = {                                                                 # Palviainen & Finer, 2012 Eur J For Res 131:945-964, eq 2, Table 7
        'N':{'Pine':[1.856,0.631,0.050],'Spruce':[2.864,0.557,0.051],'Birch':[1.590,0.788,0.044]},
        'P':{'Pine':[-0.202,0.566,0.115],'Spruce':[1.109,0.461,0.051],'Birch':[-1.034,0.838,0.162]},
        'K':{'Pine':[0.839,0.650,0.078],'Spruce':[2.487,0.478,0.061],'Birch':[-0.811,1.087,0.055]}        
        }

    #litter_nut={'N':12.0, 'P':1.0, 'K':3.5}      #mg/g Laiho 1997 page 49
    litter_nut={'Pine':{'N':6.08, 'P':0.63, 'K':2.29},      #mg/g Palviainen & Finer 2012
                'Spruce':{'N':6.9, 'P':0.81, 'K':2.81},
                'Birch':{'N':9.7, 'P':0.86, 'K':3.08}}
    
    #retrans = {'N': 0.69, 'P': 0.73, 'K':0.8}     #Nieminen Helmisaari 1996 Tree Phys
    retrans = {'N': 0.69, 'P': 0.73, 'K':0.84}
    
    litterN = litter_mass*(1.-retrans['N'])*litter_nut[species]['N']/1000.
    litterP = litter_mass*(1.-retrans['P'])*litter_nut[species]['P']/1000.
    litterK = litter_mass*(1.-retrans['K'])*litter_nut[species]['K']/1000.
    
    gvNup = np.minimum(gvNup, gvshare*(Nrel))
    gvPup = np.minimum(gvPup, gvshare*(Prel))
    gvKup = np.minimum(gvKup, gvshare*(Krel))
    
    litterN = np.minimum(litterN, littershare*(Nrel))
    litterP = np.minimum(litterP, littershare*(Prel))
    litterK = np.minimum(litterK, littershare*(Krel))

    #print ('nut bal ***************************')
    #print (np.mean(Krel), np.mean(litterK), np.mean(gvKup)) 
    #print ('****************************')

    #******** Nutrients in the stand **************    
    lna,b,k=par['N'][species]                                               
    Ns= MarjoNut(vol_ini, lna, b, k) 
    lna,b,k=par['P'][species]                                                
    Ps= MarjoNut(vol_ini, lna, b, k) 
    lna,b,k=par['K'][species]                                                
    Ks= MarjoNut(vol_ini, lna, b, k) 

    lna,b,k=par['N'][species]                                                        
    volN = MarjoBck(Ns + Nrel - litterN - gvNup, lna, b, k)
    lna,b,k=par['P'][species]                                                        
    volP = MarjoBck(Ps + Prel - litterP - gvPup, lna, b, k)
    lna,b,k=par['K'][species]                                                        
    volK = MarjoBck(Ks + Krel - litterK - gvKup, lna, b, k)
    
    volN = np.maximum(volN, vol_ini)
    volP = np.maximum(volP, vol_ini)
    volK = np.maximum(volK, vol_ini)

    return volN, volP, volK


def nutrient_demand(vol_ini, vol, litter_mass):
    #vol as list: start, end, leaf_mass kg/ha at the end, yrs is length of the study period
    #this is not in use
    
    MarjoNut = lambda vol, lna, b, k: np.exp(lna + b*np.log(vol) + k)
    from_abovegr_to_total = 1.2    

    litter_nut={'N':12.0, 'P':1.0, 'K':3.5}      #mg/g Laiho 1997 page 49
    #litter_nut={'N':6.9, 'P':0.81, 'K':2.81}      #mg/g Palviainen & Finer 2012
    retrans = {'N': 0.69, 'P': 0.73, 'K':0.8}     #Nieminen Helmisaari 1996 Tree Phys
    #longevityLeaves = {'pine':4., 'spruce':5.} #yrs, Lamppu & Huttunen 2001 CJFR life span of leaves and fine roots
    longevityLeaves = {'pine':3., 'spruce':4.} #yrs, Lamppu & Huttunen 2001 CJFR life span of leaves and fine roots

    par = {                                                                 # Palviainen & Finer, 2012 Eur J For Res 131:945-964, eq 2, Table 7
    'N':{'pine':[1.856,0.631,0.050]},
    'P':{'pine':[-2.387,0.754,0.158]},
    'K':{'pine':[0.839,0.650,0.070]}        
    }
    
    lna,b,k=par['N']['pine']                                                # pine
    N = (MarjoNut(vol, lna, b, k) - MarjoNut(vol_ini, lna, b, k))* from_abovegr_to_total + \
        litter_mass*(1.-retrans['N'])*litter_nut['N']/1000.

    lna,b,k=par['P']['pine']                                                # pine
    P = (MarjoNut(vol, lna, b, k) - MarjoNut(vol_ini, lna, b, k))* from_abovegr_to_total + \
        litter_mass*(1.-retrans['P'])*litter_nut['P']/1000.

    lna,b,k=par['K']['pine']                                                # pine
    K = (MarjoNut(vol, lna, b, k) - MarjoNut(vol_ini, lna, b, k))* from_abovegr_to_total + \
        litter_mass*(1.-retrans['K'])*litter_nut['P']/1000.


    return N, P, K



def motti_development(spara, a_arr, ifile):
    """
    Input:
        spara contains information about tree species in the stand
        array of stand age for each timestep, age in yrs
        Motti-input file name including the folder path
    Out:
        arrays of dominant height, LAI, volume in each timestep
        interpolation functions: 
            biomass to volume
            N to volume
            P to volume
            K to volume
    """
    species_codes ={1:'Pine', 2:'Spruce', 3:'Birch'}
    df, sp = get_motti(ifile, return_species_number=True)
    spe = spara['species']

    ## added test for -1 returned by get_motti, with pass
    ## note: what if sp == 3 ?
    if sp < 0:
        pass
    elif sp < 3 and (spe != species_codes[sp]):
        print('Species mismatch: check Motti simulation and species code in stand parameters')
        print('Species code in ' +  ifile +  'is ' + str(sp))
        print('Species in parameter file is ' + spe)
    elif sp > 3 and ((spe == 'Pine') or (spe=='Spruce')):
        print('Species mismatch: check Motti simulation and species code in stand parameters')
        print('Species code in ' +  ifile +  'is ' + str(sp))
        print('Species in parameter file is ' + spe)
        
    #---Nutrient concentrations in tree biomass components:
    # Palviainen & Finer 2012 Eur J For Res 131: 945-964
    #---Concentrations in mg/g
    nuts = {'Pine':{
                'Stem':{'N':1.17, 'P':0.08, 'K':0.45},
                'Crown':{'N':6.08, 'P':0.63, 'K':2.29}},
            'Spruce':{
                'Stem':{'N':1.12, 'P':0.09, 'K':0.64},
                'Crown':{'N':6.9, 'P':0.81, 'K':2.81}},
            'Birch':{
                'Stem':{'N':1.51, 'P':0.15, 'K':0.58},
                'Crown':{'N':9.7, 'P':0.86, 'K':3.08}
                    }
            }

    # Nieminen Helmisaari 1996 Tree Phys
    retrans = {'N': 0.69, 'P': 0.73, 'K':0.8}
    #yrs, life span of leaves and fine roots
    #longevityLeaves = {'Pine':4., 'Spruce':5., 'Birch':1.}
    # yrs, life span of leaves and fine roots
    longevityLeaves = {'Pine':3.0, 'Spruce':4.0, 'Birch':1.0}
    # Yuan & Chen 2010, turnover 1.07 times per year
    longevityFineRoots = {'Pine':1.0, 'Spruce':1.0, 'Birch':1.0}
    # Pine Mäkinen 1999
    longevityBranch = {'Pine':22.0, 'Spruce':22.0, 'Birch':22.0}
    
    #create interpolation functions to follow Motti-defined framework of forest development
    x = np.insert(df['age'].values, 0, 0.)
    h = np.insert(df['hdom'].values, 0, 0.)
    ageToHdom = interp1d(x, h, fill_value='extrapolate')

    #Härkönen et al. 2015 BER 20, 181-195
    sla = {'Pine': 6.8, 'Spruce': 7.25, 'Birch':14.0}

    leaf = np.insert(df['leaves'].values/10.0 * sla[spe], 0, 0.0)
    ageToLAI = interp1d(x,leaf,fill_value='extrapolate')

    yi = np.insert(df['yield'].values, 0, 0.0)
    ageToYield = interp1d(x,yi,fill_value='extrapolate')
    
    v = np.insert(df['vol'].values, 0, 0.0)
    ageToVol = interp1d(x,v,fill_value='extrapolate')

    ba = np.insert(df['BA'].values, 0, 0.0)
    ageToBa = interp1d(x,ba,fill_value='extrapolate')
    
    #create interpolation functions to follow Motti-defined framework of forest development
    logs = np.insert(df['logs'].values, 0, 0.0)   
    volToLogs = interp1d(v,logs,fill_value='extrapolate')

    p = np.insert(df['pulp'].values, 0, 0.0)
    volToPulp = interp1d(v,p,fill_value='extrapolate')


    rho = {'Pine': 400.0, 'Spruce': 380.0, 'Birch':480.0} #kg/m3

    #unit from tn/ha to kg/ha
    bm = rho[spe]*df['yield'].values + (df['branch_living'].values +
                                        df['branch_dead'].values +
                                        df['leaves'].values +
                                        df['stump'].values +
                                        df['roots_coarse'].values +
                                        df['roots_fine'].values)*1000.0
    bm = np.insert(bm, 0, 0.0)
    ageToBm = interp1d(x,bm, fill_value='extrapolate')
    bmToYi= interp1d(bm,yi, fill_value='extrapolate')
    bmToBa= interp1d(bm,ba, fill_value='extrapolate')
    
    yiToVol = interp1d(yi,v, fill_value='extrapolate')
    
    lmass =  np.insert(df['leaves'].values, 0, 0.)  #tn / ha
    bmToLeafMass = interp1d(bm, lmass, fill_value='extrapolate')
    bmToLAI = interp1d(bm, lmass* sla[spe]/10., fill_value='extrapolate')
    bmToHdom = interp1d(bm,h, fill_value='extrapolate')
    yiToBm = interp1d(yi, bm, fill_value='extrapolate')
    stems = np.insert(df['N'].values, 0, df['N'].values[0])
    bmToStems =interp1d(bm,stems, fill_value='extrapolate')
    ageToStems = interp1d(x,stems, fill_value='extrapolate')

    #stemlike biomass fraction for nutrient computation kg/ha
    bmStemlike = rho[spe]*df['yield'].values + (df['stump'].values +
                                                df['roots_coarse'].values)*1000.0
    #crownlike biomass components for nutrient computation kg/ha
    bmCrownlike = (df['branch_living'].values + df['branch_dead'].values + 
                   df['leaves'].values +  df['roots_fine'].values)*1000.0
    bmStemlike = np.insert(bmStemlike,0,0.0)
    bmCrownlike = np.insert(bmCrownlike,0,0.0)
    ageToStemlike = interp1d(x,bmStemlike, fill_value='extrapolate')
    ageToCrownlike = interp1d(x,bmCrownlike, fill_value='extrapolate')
    
    #bmLeavesFineRoots = (df['leaves'].values +  df['roots_fine'].values)*1000.
    #bmLeavesFineRoots=np.insert(bmLeavesFineRoots, 0, 0.)
    #ageToLeavesFineRoots=interp1d(x, bmLeavesFineRoots, fill_value ='extrapolate')
    #litter = ageToLeavesFineRoots(a_arr)/longevityLeaves[spe]*np.gradient(a_arr)  #Litterfall kg/ha in timestep
    #litter2 = litter.copy()
    #litter2 = np.insert(litter2, 0, 0.)
    #bm2 = ageToBm(a_arr)
    #bm2 = np.insert(bm2, 0, 0.)
    #bmToLitter = interp1d(bm2, litter2, fill_value='extrapolate')

    bmLeaves = (df['leaves'].values +  df['roots_fine'].values)*1000.0
    bmFineRoots = (df['roots_fine'].values)*1000.0
    bmLeaves = np.insert(bmLeaves, 0, 0.0)
    bmFineRoots = np.insert(bmFineRoots, 0, 0.0)
    bmBranch = (df['branch_living'].values)*1000.0
    bmBranch = np.insert(bmBranch, 0, 0.0)

    ageToLeaves = interp1d(x, bmLeaves, fill_value ='extrapolate')
    ageToFineRoots = interp1d(x, bmFineRoots, fill_value ='extrapolate')
    ageToBranch = interp1d(x, bmBranch, fill_value ='extrapolate')
    litter = ageToLeaves(a_arr)/longevityLeaves[spe]*np.gradient(a_arr) + \
             ageToFineRoots(a_arr)/longevityFineRoots[spe]*np.gradient(a_arr) + \
                 ageToBranch(a_arr)/longevityBranch[spe]*np.gradient(a_arr)    #Litterfall kg/ha in timestep
    litter2 = litter.copy()
    litter2 = np.insert(litter2, 0, 0.)
    bm2 = ageToBm(a_arr)
    bm2 = np.insert(bm2, 0, 0.)
    bmToLitter = interp1d(bm2, litter2, fill_value='extrapolate')
    #'N':12.0, 'P':1.0, 'K':3.5    
    
    N = ageToStemlike(a_arr)*nuts[spe]['Stem']['N']*1e-3 + ageToCrownlike(a_arr)*nuts[spe]['Crown']['N']*1e-3  #kg
    P = ageToStemlike(a_arr)*nuts[spe]['Stem']['P']*1e-3 + ageToCrownlike(a_arr)*nuts[spe]['Crown']['P']*1e-3
    K = ageToStemlike(a_arr)*nuts[spe]['Stem']['K']*1e-3 + ageToCrownlike(a_arr)*nuts[spe]['Crown']['K']*1e-3
    Nlit = np.cumsum(litter)*nuts[spe]['Crown']['N']*1e-3 *(1.-retrans['N']) 
    Plit = np.cumsum(litter)*nuts[spe]['Crown']['P']*1e-3 *(1.-retrans['P'])
    Klit = np.cumsum(litter)*nuts[spe]['Crown']['K']*1e-3 *(1.-retrans['K']) 

    N_demand =Nlit[-1] + N[-1]-N[0]
    P_demand =Plit[-1] + P[-1]-P[0]
    K_demand =Klit[-1] + K[-1]-K[0]
    print ('    + Stand nutrient demand')
    fill = ('      -')
    print (fill, 'Vol as yield', np.round(ageToYield(a_arr[0]),2), np.round(ageToYield(a_arr[-1]),2))
    print (fill, 'N demand', np.round(N_demand,2))
    print (fill, 'P demand', np.round(P_demand,2))
    print (fill, 'K demand', np.round(K_demand,2))

    growth =ageToYield(a_arr)-ageToYield(a_arr[0])
    #NtoVol = interp1d(N-N[0]+Nlit,growth , fill_value='extrapolate')
    #PtoVol = interp1d(P-P[0]+Plit, growth, fill_value='extrapolate')
    #KtoVol = interp1d(K-K[0]+Klit, growth, fill_value='extrapolate')

    #Nconc,Pconc,Kconc = np.mean(N/ageToBm(a_arr)),np.mean(P/ageToBm(a_arr)),np.mean(K/ageToBm(a_arr))


    return ageToHdom(a_arr), ageToLAI(a_arr), ageToVol(a_arr), ageToYield(a_arr), ageToBm(a_arr), ageToBa(a_arr), \
            ageToStems(a_arr), bmToLeafMass, bmToLAI, bmToHdom, bmToYi, bmToBa, yiToVol, yiToBm, ageToVol, bmToLitter, bmToStems, volToLogs, \
            volToPulp, sp, N_demand, P_demand, K_demand
    

#process_Motti()
def rew(dwt):
    '''Alla hyväksi havaitut'''
    wt=np.array([-150.0, -1.0, -0.5, -0.3, -0.1, 0.0])   #water table in m
    re=np.array([0.0, 0.2, 0.4 ,1.0, 1.0, 0.7])    #relative water uptake

    frew=interp1d(wt,re,fill_value='extrapolate')    
    return frew(dwt)


def rewFloor(dwt,LAI):          # Leenan testi
    # Restriction for efloor: similar to rew for transpiration but the restriction increases with decreasing LAI (3...1.5)
    LAIarray=np.array([0, 1.5, 3, 5])
    rajKerroin=np.array([1, 1, 1, 1]) 
    
    fLAI = interp1d(LAIarray,rajKerroin,fill_value='extrapolate')  
    
    wt=np.array([-150.0, -1.0, -0.5, -0.3, -0.1, 0.0])   #water table in m
    re=fLAI(LAI) * np.array([1, 1, 1, 1, 1, 1])
    
    frew=interp1d(wt,re,fill_value='extrapolate')   
    
    return frew(dwt)

def rew_drylimit(dwt):
    # Koivusalo et al. 2008 HESS without wet side limit
    wt=np.array([-150.0, -1.2, -0.7, -0.15, 0.0])   #water table in m
    re=np.array([0.0, 0.5, 1.0, 1.0, 1.0])    #relative water uptake
    

    frew=interp1d(wt,re,fill_value='extrapolate')
    return frew(dwt)

    

def assimilation(ppara, rg, vpd, Ta_minus1,Ta, rew, LAI, Xk, Ns, Ps, Ks, hdom):
    """
    Computes photosynthesis and respiration of the stand in daily time step
    Mäkelä et al. 2008. Empirical model of stand GPP LUE approach. Global Change Biology 14: 92-108
    IN:
        (develpment of all sided leaf area index (LAI) m2 m-2)
        meteorological input data:
        rg in W/m2
        vpd in kPa
        Ta previous day tamperature
        Ta  in deg C
        Xk acclimatisation temperature, Ta in the first time step
        Ns,Ps,Ks nutrient release in time step kg/ha/timestep
        Nconc, Pconc, Kconc nutrient concentration in stand 
        hdom dominant height of the stand (used in computation of respiration)
    OUT:
        Net primary production NPP in kg/ha/yr        
        Xa
    """
    par = rg*4.6*0.5 /1000000.*86400.                                          # Unit conversion to mol/m2/day, 0.5 is the share of par from rg 

    """ Eq 2 """
    fL = 1./(ppara['gamma']*par+1.)                                             #

    """Eq. 3a and 3b"""
    Xk = Xk+(Ta-Xk)/ppara['tau']    
    Sk = np.maximum(Xk-ppara['X0'], 0.0)                           

    """Eq 4"""
    fs = np.minimum(Sk/ppara['Smax'], 1.0)                         # Final temperature adustment

    """Eq 5"""                                                                  # Vapor pressure deficit function: limits photos when stomata are closed
    fd = np.exp(ppara['kappa']*vpd)                                             # Eq 5

    """Eq 6"""                                                                  # Soil water content modifyer 
    fREW=1./((1.+ ((1.-rew)/ppara['alfa'])**ppara['nu']))

    """ Beer-Lambert function: LAI function"""                                  # Fabrika 2013
    kext = 0.2
    
    """Eq 1"""
    #LAI = LAI*11./6.4     #to all sided LAI Mäkelä et al. 2001 (LAI 11)
    Pk=ppara['beta'] *(1.- np.exp(-kext*LAI)) * par * fL * fs * fd * fREW   #canopy GPP gC m-2 day-1
    """Respi"""
    #Conversion micro mol/m2/s ->g/m2/day
    #Rtrees= np.maximum((r0*q10**((Ta-10.)/10.)-cr)*rlai, np.zeros(sim_len))
    #Rtrees = 0.47*Pk  #Waring et al. 1998, Tree Physiology 18, 129-132
    fr=interp1d(np.array([0.0,10.0,20.0,30.0,]), 
                np.array([0.6, 0.50,0.36,0.20]), fill_value='extrapolate')  #Mäkelä & Valentine 2001 Tree Physiology 21, 1015-1030, Härkönen et al. 2010 ForEco 259, 524-533    

    NPP = Pk*fr(hdom)* 2.      #Change from gC m-2 to g organic matter
    
    
    return NPP*10.    #kg/ha organic matter


def assimilation_yr(ppara, dfforc, wt, afp, leaf_mass, hdom, species = 'Pine'):

    """
    Computes photosynthesis and respiration of the stand in daily time step
    Mäkelä et al. 2008. Empirical model of stand GPP LUE approach. Global Change Biology 14: 92-108
    IN:
        (develpment of all sided leaf area index (LAI) m2 m-2)
        meteorological input data:
        rg in W/m2
        vpd in kPa
        Ta previous day tamperature
        Ta  in deg C
        Xk acclimatisation temperature, Ta in the first time step
        Ns,Ps,Ks nutrient release in time step kg/ha/timestep
        Nconc, Pconc, Kconc nutrient concentration in stand 
        hdom dominant height of the stand (used in computation of respiration)
        wt water table [m] negative down, as pandas dataframe shape (days, nodes)
    OUT:
        Net primary production NPP in kg/ha/yr        
        Xa
    """
    par = np.zeros(366)
    rg, vpd, Ta = dfforc['Rg'],dfforc['vpd'],dfforc['T']    
    par[:len(rg)] = rg*4.6*0.5 /1000000.*86400.                                          # Unit conversion to mol/m2/day, 0.5 is the share of par from rg 

    """ Eq 2 """
    fL = np.zeros(366)
    fL[:len(par)] = 1./(ppara['gamma']*par+1.)                                             #

    """Eq. 3a and 3b"""
    sim_len = len(Ta)    
    Xk = np.zeros(sim_len)                                                      # Acclimatiosation: Temerature adjustment with delay tau
    Xk[0]=Ta[0]
    for n,T in enumerate(Ta[0:-1]):
        Xk[n+1] = Xk[n]+(T-Xk[n])/ppara['tau']
    Sk = np.maximum(Xk-ppara['X0'], np.zeros(sim_len))                           

    """Eq 4"""
    fs = np.zeros(366)
    fs[:len(Sk)] = np.minimum(Sk/ppara['Smax'], 1.0)                         # Final temperature adustment
    
    """Eq 5"""
    fd = np.ones(366)                                                                  # Vapor pressure deficit function: limits photos when stomata are closed
    fd[:len(vpd)] = np.exp(ppara['kappa']*vpd)                                             # Eq 5
    

    #************* From here on: different in different columns********************************
    """Eq 6"""                                                                  # Soil water content modifyer 
    rew = rew_drylimit(wt)
    days, n = np.shape(rew)
    rewarr=np.zeros((366,n))    
    rewarr[:days,:]=rew

    """air filled porosity function"""
    
    afptmp=np.array([1.0, 0.06, 0.0])   #afp in root zone
    afp_response=np.array([1.0, 1.0, 0.0])    #relative water uptake
    fafp = interp1d(afptmp,afp_response,fill_value='extrapolate')
    f_fafp = fafp(afp)
    fafparr=np.zeros((366,n))    
    fafparr[:days,:]=f_fafp


    #LAI = LAI*11./6.4     #to all sided LAI Mäkelä et al. 2001 (LAI 11)
    sla= {'Pine': (12.5+15.)/2., 'Spruce': (8.+11.)/2., 'Birch':28.}             #Härkönen et al. 2015 BER 20, 181-195, Table 3, All sided LAI      
    
    LAI = leaf_mass/10.*sla[species]       
    LAI = LAI*np.ones(n)    
    npp_arr = np.zeros(n)
    npp_arr_pot = np.zeros(n)
    
    for column in range(n):     
        fREW=((1.+ ((1.-rewarr[:,column])/ppara['alfa'])**ppara['nu']))**(-1.)
        fAFP = fafparr[:,column]
        """ Beer-Lambert function: LAI function"""                                  # Fabrika 2013
        kext = 0.2
        
        """Eq 1"""
        Pk=ppara['beta'] *(1.- np.exp(-kext*LAI[column])) * par * fL * fs * fd * fREW * fAFP   #canopy GPP gC m-2 day-1
        #Pk=ppara['beta'] *(1.- np.exp(-kext*LAI[column])) * par * fL * fs * fd * fREW    #canopy GPP gC m-2 day-1
        Pk_pot=ppara['beta'] *(1.- np.exp(-kext*LAI[column])) * par * fL * fs * fd * 1. * 1.   #growth that is not restricted by edaphic factors gC m-2 day-1

        
        """Respi"""
        #Conversion micro mol/m2/s ->g/m2/day
        fr=interp1d(np.array([0.0,10.0,20.0,30.0,]), 
                    np.array([0.6, 0.50,0.36,0.20]), fill_value='extrapolate')  #Mäkelä & Valentine 2001 Tree Physiology 21, 1015-1030, Härkönen et al. 2010 ForEco 259, 524-533    
        NPP = Pk*0.5*2.
        NPP_pot = Pk_pot*0.5*2.
        #NPP = Pk*fr(hdom[column])* 2.      #Change from gC m-2 to g organic matter
        #NPP_pot = Pk_pot*fr(hdom[column])* 2.      #Change from gC m-2 to g organic matter
        npp_arr[column] = sum(NPP)
        npp_arr_pot[column] = sum(NPP_pot)
        
    #return npp_arr*10.    #kg/ha organic matter
    return npp_arr*10., npp_arr_pot*10.    #kg/ha organic matter

def heterotrophic_respiration_yr(t5, yr, dfwt, dfair_r, v, spara):
        """
        Output: 
            mean time series kg CO2 ha-1 day-1 and annual sum for each computation  node 
        """
        sfc = spara['sfc']        
        bd_d = {2: 0.14, 3: 0.11, 4: 0.10, 5: 0.08}           # Mese study: bulk densities in different fertility classes                                                                 # peat layer thickness, cm            
        
        if spara['bd top'] is None:
            bd = bd_d[sfc]
        else:
            bd = np.mean(spara['bd top'])
        air_ratio = dfair_r[str(yr)]
        wt = dfwt[str(yr)+'-05-01':str(yr)+'-10-31'].mean().values*-100.     #.values[:-1])
        
        T5 = t5[str(yr)][0]
        
        B = 350. ; T5zero = -46.02; T5ref = 10.
        T5 = np.minimum(T5, 16.)
        Rref = (0.0695+3.7*10**(-4)*v +5.4*10**(-4)*bd + 1.2*10**(-3)*wt)*24.    #unit g/m2/h CO2-> g/m2/day
        Rhet = [rref*np.exp(B*((1./(T5ref-T5zero))-(1./(T5-T5zero)))) for rref in Rref]          
        Rhet = np.array(Rhet).T
        Rhet_root = Rhet*air_ratio              
        n, days = np.shape(Rhet)
    
        return days, np.mean(Rhet, axis=1)*10., np.sum(Rhet, axis=0)*10., np.sum(Rhet_root, axis=0)*10.       

def heterotrophic_respiration_yr_bck(forc, yr, dfwt, dfair_r, v, spara):
        """
        Output: 
            mean time series kg CO2 ha-1 day-1 and annual sum for each computation  node 
        """
        sfc = spara['sfc']        
        bd_d = {2: 0.14, 3: 0.11, 4: 0.10, 5: 0.08}           # Mese study: bulk densities in different fertility classes                                                                 # peat layer thickness, cm            
        
        if spara['bd top'] is None:
            bd = bd_d[sfc]
        else:
            bd = np.mean(spara['bd top'])
        air_ratio = dfair_r[str(yr)]
        wt = dfwt[str(yr)+'-05-01':str(yr)+'-10-31'].mean().values*-100.     #.values[:-1])
        
        T5 = forc[str(yr)]['T']
        
        B = 350. ; T5zero = -46.02; T5ref = 10.
        T5 = np.minimum(T5, 16.)
        Rref = (0.0695+3.7*10**(-4)*v +5.4*10**(-4)*bd + 1.2*10**(-3)*wt)*24.    #unit g/m2/h CO2-> g/m2/day
        Rhet = [rref*np.exp(B*((1./(T5ref-T5zero))-(1./(T5-T5zero)))) for rref in Rref]          
        Rhet = np.array(Rhet).T
        Rhet_root = Rhet*air_ratio              
        n, days = np.shape(Rhet)
    
        return days, np.mean(Rhet, axis=1)*10., np.sum(Rhet, axis=0)*10., np.sum(Rhet_root, axis=0)*10.       

def understory_uptake(n, lat, lon, barea0, barea1, stems0, stems1, yi0, yi1, sp, ts, simtime, sfc, standAge):
    """
    INPUT: 
        n - number of computation nodes between ditches
        lat - latitude in [units]
        lon - longitude in [units]
        barea 0,1 - stand basal area [m2], start, end
        stems 0,1 - number of stems per ha, start, end
        yi 0,1 - stand volume as time series, start, end
        sp - dominant tree species - 1 pine, 2 spruce, 3 birch
        ts - temperature sum [dd]
        simtime  - simulation time in yrs
        sfc - site fertility class, 
        standAge in yrs
    SOURCE:
        Muukkonen & Makipaa, 2006. Bor.Env.Res. 11, 355-369.\n
    AUTHOR:
        Samuli Launiainen 18.06.2014, Modified for array operations by Ari Laurén 13.4.2020 \n
    NOTE:
         Multi-regression models not yet tested!
         In model equations independent variables named differently to M&M (2006): here x[0] = z1, x[1]=z2, ... x[7]=z8 and x[8]=z10\n
         \n

         Site nutrient fertility class (sfc) at mires:\n
             1: herb-rich hw-spruce swamps, pine mires, fens, 
             2: V.myrtillus / tall sedge spruce swamps, tall sedge pine fens, tall sedge fens,
             3: Carex clobularis / V.vitis-idaea swamps, Carex globularis pine swamps, low sedge (oligotrophic) fens,
             4: Low sedge, dwarf-shrub & cottongrass pine bogs, ombo-oligotrophic bogs,
             5: S.fuscum pine bogs, ombotrophic and S.fuscum low sedge bogs.
         Drainage status x[8] at mires (Paavilainen & Paivanen, 1995):
             1: undrained
             2: Recently draines, slight effect on understory veg., no effect on stand
             3: Transforming drained mires, clear effect on understory veg and stand
             4: Transforming drained mires, veget. resembles upland forest site type, tree-stand forest-like.
  
    """
    #ATTN! convert barea, stems, yi, standAge from time series to list containing start and end state (adjustment to annual call)

    if sp == 2: 
        smc = np.ones(n)*2      #spruce
    else:
        smc = np.ones(n)*3      #pine and others
    
    #n, lat, lon, barea, stems, yi, sp, ts, simtime, sfc     
    age = np.ones(n) * standAge                                                
    sfc = np.ones(n) * sfc  ## x2 surface elevation m asl
    dem = np.ones(n) * 80.  ## x2 surface elevation m asl
   
    vol = yi0 #np.ones(n) * yi[0]  # x5 stand volume m3 ha-1
    ba = barea0 #np.ones(n) * barea[0]  # x7 basal area m2 ha-1    
    Nstems = stems0 #np.ones(n)*stems[0]   # x6 number of stems -ha, default 900
    
    #expected_yield = np.ones(n)*(yi[-1]- yi[0])
    
    #------------- classify and map pixels-------------------------------------------------------- 
    ix_spruce_mire = np.where(np.equal(smc, 2))
    ix_pine_bog = np.where(np.equal(smc, 3))
    ix_open_peat = np.where(np.equal(smc, 4))
    
    #---------------------------------------
    #ATTN: change here a module from NutSpaFHy (Auran tekemä muokkaus)
    #latitude = 0.0897*lat/10000. + 0.3462                                       #approximate conversion to decimal degrees within Finland,  N
    #longitude = 0.1986*(lon-3000000)/10000. + 17.117                            #approximate conversion to decimal degrees within Finland in degrees E
    
    inProj = Proj(init='epsg:3067')
    outProj = Proj(init='epsg:4326')
    longitude,latitude = transform(inProj,outProj,lon,lat)

    drain_s = 4      # x8 drainage status, default value 4

    #---------------------------------------

    def gv_biomass_and_nutrients(n, ix_spruce_mire, ix_pine_bog,
                ix_open_peat, latitude, longitude, dem, ts, sfc, vol, Nstems, ba, drain_s, age):   
        #--------------- nutrient contents in vegetation-----------------------
        """
        Computed:
           - total biomass and bottom layer; field layer is gained as a difference of tot and bottom layer (more cohrent results)
           - N and P storage in the each pixel
           - annual use of N and P due to litterfall
        Muukkonen Mäkipää 2005 upland sites: field layer contains dwarf shrubs and (herbs + grasses), see Fig 1
            share     dwarf shrubs     herbs 
            - Pine       91%            9%
            - Spruce     71%            29%
            - broad l    38%            62%
        Peatland sites (assumption):
            share      dwarf shrubs    herbs
            - Pine bogs    90%          10%
            - Spruce mires 50%          50%
        Palviainen et al. 2005 Ecol Res (2005) 20: 652–660, Table 2
        Nutrient concentrations for
                                N              P           K
            - Dwarf shrubs      1.2%         1.0 mg/g     4.7 mg/g
            - herbs & grasses   1.8%         2.0 mg/g    15.1 mg/g
            - upland mosses     1.25%        1.4 mg/g     4.3 mg/g
        Nutrient concentrations for sphagna (FIND):
                                N              P     for N :(Bragazza et al Global Change Biology (2005) 11, 106–114, doi: 10.1111/j.1365-2486.2004.00886.x)
            - sphagnum          0.6%           1.4 mg/g     (Palviainen et al 2005)   
        Annual litterfall proportions from above-ground biomass (Mälkönen 1974, Tamm 1953):
            - Dwarf shrubs          0.2
            - herbs & grasses        1
            - mosses                0.3
            Tamm, C.O. 1953. Growth, yield and nutrition in carpets of a forest moss (Hylocomium splendens). Meddelanden från Statens Skogsforsknings Institute 43 (1): 1-140.
        We assume retranslocation of N and P away from senescing tissues before litterfall:
                                N           P
            - Dwarf shrubs     0.5         0.5
            - Herbs & grasses  0.5         0.5
            - mossess          0.0         0.0
        
        Turnover of total biomass including the belowground biomass is assumed to be 1.2 x above-ground biomass turnover
        
        """

        fl_share = {'description': 'share of dwarf shrubs (ds) and herbas & grasses (h) from field layer biomass, kg kg-1',
                    'pine_upland':{'ds': 0.91, 'h': 0.09}, 'spruce_upland':{'ds': 0.71, 'h': 0.29}, 
                    'broadleaved_upland':{'ds': 0.38, 'h': 0.62}, 'spruce_mire':{'ds': 0.50, 'h': 0.50}, 
                    'pine_bog':{'ds': 0.90, 'h': 0.10}}
        nut_con ={'description': 'nutrient concentration of dwarf shrubs (ds), herbs & grasses (h), upland mosses (um), and sphagna (s), unit mg/g',
                  'ds':{'N':12.0, 'P':1.0, 'K': 4.7}, 'h':{'N':18.0, 'P':2.0, 'K': 15.1}, 'um':{'N':12.5, 'P':1.4, 'K':4.3}, 
                  's':{'N':6.0, 'P':1.4, 'K':4.3}}
        lit_share = {'description': 'share of living biomass that is lost as litter annually for dwarf shrubs (ds), herbs & grasses (h), upland mosses (um), and sphagna (s), unit: kg kg-1',
                   'ds': 0.2, 'h': 0.5, 'um': 0.3, 's': 0.3}
        green_share = {'description': 'share of green mass from the total biomass of dwarf shrubs (ds), herbs & grasses (h), upland mosses (um), and sphagna (s), unit kg/kg',
                       'ds': 0.2, 'h': 0.5, 'um': 0.3, 's': 0.3}
        retrans ={'description': 'share of nutrients retranslocated before litterfallfor dwarf shrubs (ds), herbs & grasses (h), upland mosses (um), and sphagna (s), unit: kg kg-1',
                  'ds': {'N':0.69, 'P':0.73, 'K':0.87},'h': {'N':0.69, 'P':0.73, 'K':0.87}, 
                  'um': {'N':0.0, 'P':0.0, 'K':0.0},'s': {'N':0.0, 'P':0.0, 'K':0.0}}
        # retrans for pine {'N': 0.69, 'P': 0.73, 'K':0.87}
        #ATTN: changes 4.1.2021 fl_share pine_bog vs spruce_mire (vice versa)
        #check, and change back restranslocation for herbs 
        
        fl_to_total_turnover = 1.2    # converts the turnover of above-ground bionmass to total including root turnover
        fl_above_to_total = 1.7       # converts aboveground biomass to total biomass 
        
        #--------- create output arrays -----------------------------
        gv_tot = np.zeros(n)                           # Ground vegetation mass kg ha-1
        gv_field = np.zeros(n)                         # Field layer vegetation mass
        gv_bot = np.zeros(n)                           # Bottom layer vegetation mass
        gv_leafmass = np.zeros(n)                      # Leaf mass in ground vegetation kg ha-1
        ds_litterfall = np.zeros(n)                    # dwarf shrub litterfall kg ha-1 yr-1
        h_litterfall = np.zeros(n)                     # herbs and grasses litterfall kg ha-1 yr-1
        s_litterfall = np.zeros(n)                     # sphagnum mosses litterfall kg ha-1 yr-1
        nup_litter = np.zeros(n)                       # N uptake due to litterfall kg ha-1 yr-1
        pup_litter = np.zeros(n)                       # P uptake due to litterfall kg ha-1 yr-1
        kup_litter = np.zeros(n)                       # K uptake due to litterfall kg ha-1 yr-1
        n_gv = np.zeros(n)                             # N in ground vegetation kg ha-1
        p_gv = np.zeros(n)                             # P in ground vegetation kg ha-1
        k_gv = np.zeros(n)                             # K in ground vegetation kg ha-1

        """------ Ground vegetation models from Muukkonen & Mäkipää 2006 BER vol 11, Tables 6,7,8"""    
        
        #     ***************** Spruce mire ***************************************
        ix = ix_spruce_mire        
        gv_tot[ix] = np.square(35.52 +0.001*longitude*dem[ix] -1.1*drain_s**2 -2e-5*vol[ix]*Nstems[ix] \
                                +4e-5*Nstems[ix]*age[ix] +0.139*longitude*drain_s) -0.5 + 116.54 # Total, Eq.39, Table 9
        gv_bot[ix] =  np.square(-3.182 + 0.022*latitude*longitude +2e-4*dem[ix]*age[ix] \
                                -0.077*sfc[ix]*longitude -0.003*longitude*vol[ix] + 2e-4*np.square(vol[ix]))-0.5 + 98.10  #Bottom layer total, Eq. 35, Table 9
        gv_field[ix] =  np.square(23.24 -1.163*drain_s**2 +1.515*sfc[ix]*drain_s -2e-5*vol[ix]*Nstems[ix]\
                                +8e-5*ts*age[ix] +1e-5*Nstems[ix]*dem[ix])-0.5 +  162.58   #Field layer total, Eq. 37, Table 9
        
       # removing inconsistent values
        gv_field[ix] = np.minimum(gv_tot[ix], gv_field[ix])
        gv_bot[ix] = np.minimum(gv_tot[ix], gv_bot[ix])        
        gv_field[ix] = np.maximum(gv_field[ix], gv_tot[ix] - gv_bot[ix])

        #annual litterfall rates
        #ATTN! vaihda nämä suoraan field layeriksi, poista tot ja bommomin kautta menevä yhteys
        ds_litterfall[ix] = fl_share['spruce_mire']['ds'] * (gv_tot[ix]-gv_bot[ix])*lit_share['ds'] * fl_to_total_turnover
        h_litterfall[ix]  = fl_share['spruce_mire']['h'] * (gv_tot[ix]-gv_bot[ix])*lit_share['h'] * fl_to_total_turnover
        s_litterfall[ix]  = gv_bot[ix]*lit_share['s']
        
        #ATTN! tarkista tämä, onko järkevä? Tee oma dictionary lehtimassalle
        gv_leafmass[ix]   = fl_share['spruce_mire']['ds'] * (gv_tot[ix]-gv_bot[ix]) * green_share['ds'] + \
                               fl_share['spruce_mire']['h']*(gv_tot[ix]-gv_bot[ix]) * green_share['h'] + \
                               gv_bot[ix] * green_share['s']
        
        
        n_gv[ix] = gv_field[ix] * fl_share['spruce_mire']['ds'] * nut_con['ds']['N']*1e-3 * fl_above_to_total \
                        + gv_field[ix] * fl_share['spruce_mire']['h'] * nut_con['h']['N']*1e-3 * fl_above_to_total \
                        + gv_bot[ix] * nut_con['s']['N']*1e-3
        p_gv[ix] = gv_field[ix] * fl_share['spruce_mire']['ds'] * nut_con['ds']['P']*1e-3 * fl_above_to_total \
                        + gv_field[ix] * fl_share['spruce_mire']['h'] * nut_con['h']['P']*1e-3 * fl_above_to_total \
                        + gv_bot[ix] *nut_con['s']['P']*1e-3
        k_gv[ix] = gv_field[ix] * fl_share['spruce_mire']['ds'] * nut_con['ds']['K']*1e-3 * fl_above_to_total \
                        + gv_field[ix] * fl_share['spruce_mire']['h'] *  nut_con['h']['K']*1e-3 * fl_above_to_total \
                        + gv_bot[ix] *nut_con['s']['K']*1e-3
        
        nup_litter[ix] = ds_litterfall[ix] * nut_con['ds']['N']*1e-3 * (1.0 -retrans['ds']['N']) \
                        + h_litterfall[ix] * nut_con['h']['N']*1e-3 * (1.0 -retrans['h']['N']) \
                        + s_litterfall[ix] * nut_con['s']['N']*1e-3 * (1.0 -retrans['s']['N'])
        
        pup_litter[ix] = ds_litterfall[ix] * nut_con['ds']['P']*1e-3 * (1.0 -retrans['ds']['P']) \
                        + h_litterfall[ix] * nut_con['h']['P']*1e-3 * (1.0 -retrans['h']['P']) \
                        + s_litterfall[ix] * nut_con['s']['P']*1e-3 * (1.0 -retrans['s']['P'])
        
        kup_litter[ix] = ds_litterfall[ix] * nut_con['ds']['K']*1e-3 * (1.0 -retrans['ds']['K']) \
                        + h_litterfall[ix] * nut_con['h']['K']*1e-3 * (1.0 -retrans['h']['K']) \
                        + s_litterfall[ix] * nut_con['s']['K']*1e-3 * (1.0 -retrans['s']['K'])
        
       # ***************** Pine bogs ***************************************

        ix = ix_pine_bog            
        gv_tot[ix] =  np.square(50.098 + 0.005 * longitude*dem[ix] -1e-5 * vol[ix] * Nstems[ix] + 0.026 * sfc[ix] * age[ix] \
                      -1e-4 * dem[ix] * ts - 0.014 * vol[ix] * drain_s) - 0.5 + 167.40                #Total, Eq 45, Table 9           
        gv_bot[ix] =  np.square(31.809 + 0.008 * longitude * dem[ix] -3e-4 * Nstems[ix] * ba[ix] \
                                + 6e-5 * Nstems[ix] * age[ix] -0.188 * dem[ix]) -0.5 + 222.22                #Bottom layer total, Eq 41, Table 9
        gv_field[ix] =  np.square(48.12 - 1e-5 * ts**2 + 0.013 * sfc[ix] * age[ix] -0.04 * vol[ix] * drain_s \
                                + 0.026 * sfc[ix] * vol[ix]) - 0.5 +133.26                                        #Field layer total, Eq. 43, Table 9

        # removing inconsistent values
        gv_field[ix] = np.minimum(gv_tot[ix], gv_field[ix])
        gv_bot[ix] = np.minimum(gv_tot[ix], gv_bot[ix])        
        gv_field[ix] = np.maximum(gv_field[ix], gv_tot[ix] - gv_bot[ix])

        # annual litterfall rates

        ds_litterfall[ix] = fl_share['pine_bog']['ds'] * (gv_tot[ix]-gv_bot[ix]) * lit_share['ds'] * fl_to_total_turnover
        h_litterfall[ix]  = fl_share['pine_bog']['h']  * (gv_tot[ix]-gv_bot[ix]) * lit_share['h'] * fl_to_total_turnover
        s_litterfall[ix]  = gv_bot[ix] * lit_share['s']
        gv_leafmass[ix]   = fl_share['pine_bog']['ds'] * (gv_tot[ix]-gv_bot[ix]) * green_share['ds'] + \
                            fl_share['pine_bog']['h'] * (gv_tot[ix]-gv_bot[ix]) * green_share['h'] + \
                            gv_bot[ix]*green_share['s']
      
        
        n_gv[ix] =        gv_field[ix] * fl_share['pine_bog']['ds'] * nut_con['ds']['N']*1e-3 * fl_above_to_total \
                        + gv_field[ix] * fl_share['pine_bog']['h'] * nut_con['h']['N']*1e-3 * fl_above_to_total \
                        + gv_bot[ix] * nut_con['s']['N']*1e-3
                        
        p_gv[ix] =        gv_field[ix] * fl_share['pine_bog']['ds'] * nut_con['ds']['P']*1e-3 * fl_above_to_total \
                        + gv_field[ix] * fl_share['pine_bog']['h'] * nut_con['h']['P']*1e-3 * fl_above_to_total \
                        + gv_bot[ix] * nut_con['s']['P']*1e-3
        
        k_gv[ix] =        gv_field[ix] * fl_share['pine_bog']['ds'] * nut_con['ds']['K']*1e-3 * fl_above_to_total \
                        + gv_field[ix] * fl_share['pine_bog']['h'] * nut_con['h']['K']*1e-3 * fl_above_to_total \
                        + gv_bot[ix] * nut_con['s']['K']*1e-3
        
        nup_litter[ix] =  ds_litterfall[ix] * nut_con['ds']['N']*1e-3 * (1.0 -retrans['ds']['N']) \
                        + h_litterfall[ix] * nut_con['h']['N']*1e-3 * (1.0 -retrans['h']['N']) \
                        + s_litterfall[ix] * nut_con['s']['N']*1e-3 * (1.0 -retrans['s']['N'])
        
        pup_litter[ix] =  ds_litterfall[ix] * nut_con['ds']['P']*1e-3 * (1.0 -retrans['ds']['P']) \
                        + h_litterfall[ix] * nut_con['h']['P']*1e-3 * (1.0 -retrans['h']['P']) \
                        + s_litterfall[ix] * nut_con['s']['P']*1e-3 * (1.0 -retrans['s']['P'])
                        
        kup_litter[ix] =  ds_litterfall[ix] * nut_con['ds']['K']*1e-3 * (1.0 -retrans['ds']['K']) \
                        + h_litterfall[ix] * nut_con['h']['K']*1e-3 * (1.0 -retrans['h']['K']) \
                        + s_litterfall[ix] * nut_con['s']['K']*1e-3 * (1.0 -retrans['s']['K'])

        
        #------------Change clear-cut areas: reduce to 1/3 of modelled ---------------------------------------------------
        to_cc = 0.33
        #ix_cc = np.where(np.logical_and(gisdata['age']<5.0, gisdata['smc']!=4))  #small stands excluding open peatlands
        ix_cc = np.where(age<5.0)
        n_gv[ix_cc] = n_gv[ix_cc] * to_cc 
        p_gv[ix_cc] = p_gv[ix_cc] * to_cc
        k_gv[ix_cc] = k_gv[ix_cc] * to_cc
        nup_litter[ix_cc] = nup_litter[ix_cc] * to_cc
        pup_litter[ix_cc] = pup_litter[ix_cc] * to_cc 
        kup_litter[ix_cc] = kup_litter[ix_cc] * to_cc 
        gv_tot[ix_cc] = gv_tot[ix_cc] * to_cc

        litterfall_gv = ds_litterfall + h_litterfall + s_litterfall
        return n_gv, p_gv, k_gv, nup_litter, pup_litter, kup_litter, gv_tot, litterfall_gv, gv_leafmass
        

    # initial N and P mass, kg ha-1    
    n_gv, p_gv, k_gv, nup_litter, pup_litter, kup_litter, gv_tot, litterfall_gv, gv_leafmass = gv_biomass_and_nutrients(n,  ix_spruce_mire, ix_pine_bog,
                ix_open_peat, latitude, longitude, dem, ts, sfc, vol, Nstems, ba, drain_s, age)

 
    # ground vegetation mass at the end of simulation, kg ha-1    
    vol = yi1         #vol + expected_yield
    Nstems = stems1   #np.ones(n)*stems[-1]
    ba = barea1       #np.ones(n)*barea[-1]

    age = np.ones(n)*(standAge + simtime)
    n_gv_end, p_gv_end, k_gv_end, nup_litter_end, pup_litter_end, kup_litter_end, gv_tot_end, litterfall_gv_end, gv_leafmass_end = gv_biomass_and_nutrients(n, ix_spruce_mire, ix_pine_bog,
                ix_open_peat, latitude, longitude, dem, ts, sfc, vol, Nstems, ba, drain_s, age)
    
    # nutrient uptake due to net change in gv biomass, only positive values accepted, negative do not associate to nutrient uptake
    nup_net = np.where(n_gv_end - n_gv > 0.0, n_gv_end - n_gv, 0.0)
    pup_net = np.where(p_gv_end - p_gv > 0.0, p_gv_end - p_gv, 0.0)
    kup_net = np.where(k_gv_end - k_gv > 0.0, k_gv_end - k_gv, 0.0)
    
    nup_litter_mean = np.mean([nup_litter, nup_litter_end], axis = 0)
    pup_litter_mean = np.mean([pup_litter, pup_litter_end], axis = 0)
    kup_litter_mean = np.mean([kup_litter, kup_litter_end], axis = 0)
    
    nup = nup_net + nup_litter_mean*simtime         # total N uptake kg ha-1 simulation time (in yrs) -1
    pup = pup_net + pup_litter_mean*simtime         # total P uptake kg ha-1 simulation time (in yrs) -1
    kup = kup_net + kup_litter_mean*simtime         # total P uptake kg ha-1 simulation time (in yrs) -1
    
    litterfall_tot = np.mean([litterfall_gv, litterfall_gv_end], axis=0) * simtime  #total gv litterfall in the simulation time kg ha-1
    gv_leafmass_mean = np.mean([gv_leafmass, gv_leafmass_end], axis = 0)
    
    """
    print ('    + Ground vegetation nutrient demand')
    fill = ('      -')
    print (fill, 'gv biomass', np.round(np.mean(gv_tot),2),'...' ,np.round(np.mean(gv_tot_end),2))
    print (fill, 'N demand', np.round(np.mean(nup),2))
    print (fill, 'P demand', np.round(np.mean(pup),2))
    print (fill, 'K demand', np.round(np.mean(kup),2))
    print (fill, 'gv litterfall', np.round(np.mean(litterfall_tot),2))
    """
    return nup, pup, kup, litterfall_tot, gv_leafmass

def get_temp_sum(forc):
    base = 5.0    
    dd = forc['T']-base    
    dd[dd<base] = 0.0
    dds = dd.sum(axis=0)                                              #cumulative temperature sum degree days    
    yrs = np.shape(forc)[0] / 365.25
    ts = (dds/yrs)
    return ts
