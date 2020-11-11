# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:42:44 2020

@author: alauren
"""

import numpy as np
import pandas as pd
import datetime
import matplotlib.pylab as plt
import seaborn as sns
from scipy.interpolate import interp1d

from canopygrid import CanopyGrid
from mosslayer import MossLayer
from strip import StripHydrology, drain_depth_development

import susi_io
from susi_utils import read_FMI_weather
from susi_utils import heterotrophic_respiration_yr
from susi_utils import nutrient_release,  rew_drylimit, nutrient_demand, nut_to_vol
from susi_utils import motti_development,  get_motti, assimilation_yr
from susi_utils import get_mese_input, get_mese_out, peat_hydrol_properties, CWTr, wrc
from susi_para import get_susi_para


#%%
folderName=r'C:/Users/alauren/Documents/WinPython-64bit-2.7.10.3/Susi_8_3_py37/outputs/'
susiPath = r'C:/Users/alauren/Documents/WinPython-64bit-2.7.10.3/Susi_8_3_py37/'
wpath = r'C:/Users/alauren/Documents/WinPython-64bit-2.7.10.3/Susi_8_3_py37/wfiles/'
mottipath = r'C:/Users/alauren/Documents/WinPython-64bit-2.7.10.3/Susi_8_3_py37/motti2/'


start_date = datetime.datetime(1980,1,1); end_date=datetime.datetime(1984,12,31)
start_yr = start_date.year; end_yr = end_date.year
yrs = (end_date - start_date).days/365.25
length = (end_date - start_date).days +1

site = 'develop_scens'
#site = 'develop'
    
mottifile = mottipath + 'Viitasaari_Mtkg.xls'
df = get_motti(mottifile)

wdata  =  'Viitasaari_weather.csv' #idata.T[nro]['wfile']
forc=read_FMI_weather(0, start_date, end_date, sourcefile=wpath+wdata)           # read weather input

sfc =  2 #idata.T[nro]['sfc']                                                                         #soil fertility class
ageSim=  50. #idata.T[nro]['age_ini']                                                                     #90
sarkaSim = 40. #idata.T[nro]['stripw']  
n = int(sarkaSim / 2)
ddwest = -0.9 #-idata.T[nro]['dd_west']/100.
ddeast = -0.9 #-idata.T[nro]['dd_east']/100.
bd = 0.14 #idata.T[nro]['bd']
peatN = None #idata.T[nro]['n_mg/g']/10.  
peatP = None #idata.T[nro]['p_mg/g']/10.
peatK = None# idata.T[nro]['k_mg/g']/10.
kaista = 1 #idata.T[nro]['kaista']



wpara, cpara, org_para, spara, outpara, photopara = get_susi_para(wlocation='undefined', peat=site, 
                                                                  folderName=folderName, hdomSim=None,  
                                                                  ageSim=ageSim, sarkaSim=sarkaSim, sfc=sfc, 
                                                                  susiPath=susiPath,
                                                                  ddwest=ddwest, ddeast=ddeast, n=n, bd=bd,
                                                                  peatN=peatN, peatP=peatP, peatK=peatK)
spara['vonP']=True
#%%

nLyrs = spara['nLyrs']                                                 # number of soil layers
dz = np.ones(nLyrs)*spara['dzLyr']                                     # thickness of layers, m
z = np.cumsum(dz)-dz/2.                                                # depth of the layer center point, m 
if spara['vonP']:
    lenvp=len(spara['vonP top'])    
    vonP = np.ones(nLyrs)*spara['vonP bottom'] 
    vonP[0:lenvp] = spara['vonP top']                                      # degree of  decomposition, von Post scale
    ptype = spara['peat type bottom']*spara['nLyrs']
    lenpt = len(spara['peat type']); ptype[0:lenpt] = spara['peat type']    
    pF, Ksat = peat_hydrol_properties(vonP, var='H', ptype=ptype) # peat hydraulic properties after Päivänen 1973    


for n in range(nLyrs):
    if z[n] < 0.41: 
        Ksat[n]= Ksat[n]*spara['anisotropy']
    else:
        Ksat[n]= Ksat[n]*1.
        
hToSto, stoToGwl, hToTra, C, hToRatio, hToAfp = CWTr(nLyrs, z, dz, pF, Ksat, direction='negative') 
#%%
z = np.array(z)   
dz =np.array(dz)
nroot = 6
nroot2 = 2

#--------- Connection between gwl and water storage------------
d = -6    
gwl=np.linspace(0,d,150)
sto = [sum(wrc(pF, x = np.minimum(z+g, 0.0))*dz) for g in gwl]     #equilibrium head m
storoot = [np.sum(wrc(pF, x = np.minimum(z+g, 0.0))[0:nroot]*dz[0:nroot]) for g in gwl]
storoot2 = [np.sum(wrc(pF, x = np.minimum(z+g, 0.0))[0:nroot2]*dz[0:nroot2]) for g in gwl]
gwlToSto = interp1d(np.array(gwl), np.array(sto), fill_value='extrapolate')

k = 30
airtot = sto[0]-sto
airroot = storoot[0]-storoot
ratio = airroot[1:k]/airtot[1:k]
plt.plot(ratio, gwl[1:k], 'g-', label='Share of nutrient release in root zone')

afproot = (storoot2[0]-storoot2)/(sum(dz[:nroot2]))
#afp = afproot/gwlToSto(0.0)
plt.plot(afproot[:k], gwl[:k], 'r-', label='air-filled porosity')
plt.legend(loc='best')
plt.vlines(0.06, -1, 0)
plt.ylabel('WT [m]')





#%%
mottipath = r'C:/Users/alauren/Documents/WinPython-64bit-2.7.10.3/Susi_7_0_py27/motti2/'

mottifile = mottipath + 'Viitasaari_Mtkg.xls'
df = get_motti(mottifile)
#print (df)
length = np.shape(forc)[0]
agearray = ageSim + np.arange(0,length,1.)/365.
print (length)

hdom, LAI, vol, yi, bm, bmToLeafMass, bmToHdom, bmToYi, yiToBm,  \
    ageToVol, bmToLitter, bmToStems, volToLogs, volToPulp = motti_development(spara, agearray, mottifile)             # dom hright m, LAI m2m-2, vol m3/ha, yield m3/ha, biomass kg/ha

print (yi[0], yi[-1])
#%%
df = get_motti(mottifile, return_spe=True)
print (df)
#%%
print (spara)
print (type(spara['sfc']))
#%%
#test for understorey vegetation
def understory_uptake(n, lat, lon, ba, stems, yi, sp, ts, simtime, sfc):
    """
    Created on Wed Jun 18 12:07:47 2014

    @author: slauniai

    Computes understory biomasses using models of Muukkonen & Makipaa, 2006 Bor. Env. Res.\n
    INPUT:
        n - number of nodes in the transect
        lat - latitude in YKJ or EUREF equivalent 
        lon - longitude 
        ts - annual temperature sum in degree days 
        expected_yield of stand during the simulation period m3 ha-1
        simtime - simulation time in years
        x - array of independent variables (optional, if not provided age-based model is used):
            x[0]=lat (degN, in decimal degrees)
            x[1]=lon (degE in decimal degrees) 
            x[2]=elev (m)
            x[3]=temperature sum (degC)
            x[4]=site nutrient level (-) 
            x[5]=stem vol. (m3 ha-1)
            x[6]=stem nr (ha-1)
            x[7]=basal area (m2 ha-1)
            x[8]=site drainage status,integer
    OUTPUT:
        y - dry biomasses (kg ha-1) of different groups\n
    SOURCE:
        Muukkonen & Makipaa, 2006. Bor.Env.Res. 11, 355-369.\n
    AUTHOR:
        Samuli Launiainen 18.06.2014, Modified for array operations by Ari Laurén 13.4.2020 \n
    NOTE:
         Multi-regression models not yet tested!
         In model equations independent variables named differently to M&M (2006): here x[0] = z1, x[1]=z2, ... x[7]=z8 and x[8]=z10\n
         \n
         Site nutrient level x[4] at upland sites:
             1: herb-rich forest 
             2: herb-rich heat f. 
             3: mesic heath f. 
             4: sub-xeric heath f.
             5: xeric heath f. 
             6: barren heath f.
             7: rock,cliff or sand f. 
         Site nutrient level x[4] at mires:\n
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

    if sp == 2: 
        smc=np.ones(n)*2      #spruce
    else:
        smc=np.ones(n)*3      #pine and others
        
    age = np.ones(n)*40.
    sfc = np.ones(n)*sfc  ## x2 surface elevation m asl
    dem = np.ones(n)*80.  ## x2 surface elevation m asl
    vol = np.ones(n)*yi[0]  # x5 stand volume m3 ha-1
    ba = np.ones(n)*ba[0]  # x7 basal area m2 ha-1    
    expected_yield = np.ones(n)*(yi[-1]- yi[0])
    
    #------------- classify and map pixels-------------------------------------------------------- 
    ix_spruce_mire = np.where(np.equal(smc, 2))
    ix_pine_bog = np.where(np.equal(smc, 3))
    ix_open_peat = np.where(np.equal(smc, 4))
    
    #---------------------------------------
    latitude = 0.0897*lat/10000. + 0.3462                                       #approximate conversion to decimal degrees within Finland,  N
    longitude = 0.1986*(lon-3000000)/10000. + 17.117                            #approximate conversion to decimal degrees within Finland in degrees E
    Nstems = bmToStems(bm[0])   # x6 number of stems -ha, default 900
    drain_s =4      # x8 drainage status, default value 4

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
                    'broadleaved_upland':{'ds': 0.38, 'h': 0.62}, 'spruce_mire':{'ds': 0.90, 'h': 0.10}, 
                    'pine_bog':{'ds': 0.50, 'h': 0.50}}
        nut_con ={'description': 'nutrient concentration of dwarf shrubs (ds), herbs & grasses (h), upland mosses (um), and sphagna (s), unit mg/g',
                  'ds':{'N':12.0, 'P':1.0, 'K': 4.7}, 'h':{'N':18.0, 'P':2.0, 'K': 15.1}, 'um':{'N':12.5, 'P':1.4, 'K':4.3}, 
                  's':{'N':6.0, 'P':1.4, 'K':4.3}}
        lit_share = {'description': 'share of living biomass that is lost as litter annually for dwarf shrubs (ds), herbs & grasses (h), upland mosses (um), and sphagna (s), unit: kg kg-1',
                   'ds': 0.2, 'h': 0.5, 'um': 0.3, 's': 0.3}
        retrans ={'description': 'share of nutrients retranslocated before litterfallfor dwarf shrubs (ds), herbs & grasses (h), upland mosses (um), and sphagna (s), unit: kg kg-1',
                  'ds': {'N':0.5, 'P':0.5, 'K':0.5},'h': {'N':0.5, 'P':0.5, 'K':0.5}, 
                  'um': {'N':0.0, 'P':0.0, 'K':0.0},'s': {'N':0.0, 'P':0.0, 'K':0.0}}
        fl_to_total_turnover = 1.2   # converts the turnover of above-ground bionmass to total including root turnover
        fl_above_to_total = 1.7   # converts aboveground biomass to total biomass 
        
        #--------- create output arrays -----------------------------
        gv_tot = np.zeros(n)                           # Ground vegetation mass kg ha-1
        gv_field = np.zeros(n)                         # Field layer vegetation mass
        gv_bot = np.zeros(n)                           # Bottom layer vegetation mass
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
     
        #***************** Spruce mire ***************************************
        ix = ix_spruce_mire        
        gv_bot[ix] =  np.square(-3.182 + 0.022*latitude*longitude +2e-4*dem[ix]*age[ix] \
                                -0.077*sfc[ix]*longitude -0.003*longitude*vol[ix] + 2e-4*np.square(vol[ix]))-0.5 + 98.10  #Bottom layer total
        gv_field[ix] =  np.square(23.24 -1.163*drain_s**2 +1.515*sfc[ix]*drain_s -2e-5*vol[ix]*Nstems\
                                +8e-5*ts*age[ix] +1e-5*Nstems*dem[ix])-0.5 +  162.58   #Field layer total
        gv_tot[ix] = np.square(35.52 +0.001*longitude*dem[ix] -1.1*drain_s**2 -2e-5*vol[ix]*Nstems \
                                +4e-5*Nstems*age[ix] +0.139*longitude*drain_s) -0.5 + 116.54 #Total
        #annual litterfall rates
        ds_litterfall[ix] = fl_share['spruce_mire']['ds']*(gv_tot[ix]-gv_bot[ix])*lit_share['ds']*fl_to_total_turnover
        h_litterfall[ix] = fl_share['spruce_mire']['h']*(gv_tot[ix]-gv_bot[ix])*lit_share['h']*fl_to_total_turnover
        s_litterfall[ix] = gv_bot[ix]*lit_share['s']
        n_gv[ix] = gv_field[ix] * fl_share['spruce_mire']['ds']*nut_con['ds']['N']*1e-3*fl_above_to_total \
                        +gv_field[ix] * fl_share['spruce_mire']['h']*nut_con['h']['N']*1e-3*fl_above_to_total \
                        +gv_bot[ix] *nut_con['s']['N']*1e-3
        p_gv[ix] = gv_field[ix] * fl_share['spruce_mire']['ds']*nut_con['ds']['P']*1e-3*fl_above_to_total \
                        +gv_field[ix] * fl_share['spruce_mire']['h']*nut_con['h']['P']*1e-3*fl_above_to_total \
                        +gv_bot[ix] *nut_con['s']['P']*1e-3
        k_gv[ix] = gv_field[ix] * fl_share['spruce_mire']['ds']*nut_con['ds']['K']*1e-3*fl_above_to_total \
                        +gv_field[ix] * fl_share['spruce_mire']['h']*nut_con['h']['K']*1e-3*fl_above_to_total \
                        +gv_bot[ix] *nut_con['s']['K']*1e-3
        nup_litter[ix] = ds_litterfall[ix] * nut_con['ds']['N']*1e-3 * (1.0 -retrans['ds']['N']) \
                        +h_litterfall[ix] * nut_con['h']['N']*1e-3 * (1.0 -retrans['h']['N']) \
                        +s_litterfall[ix] * nut_con['s']['N']*1e-3 * (1.0 -retrans['s']['N'])
        pup_litter[ix] = ds_litterfall[ix] * nut_con['ds']['P']*1e-3 * (1.0 -retrans['ds']['P']) \
                        +h_litterfall[ix] * nut_con['h']['P']*1e-3 * (1.0 -retrans['h']['P']) \
                        +s_litterfall[ix] * nut_con['s']['P']*1e-3 * (1.0 -retrans['s']['P'])
        kup_litter[ix] = ds_litterfall[ix] * nut_con['ds']['K']*1e-3 * (1.0 -retrans['ds']['K']) \
                        +h_litterfall[ix] * nut_con['h']['K']*1e-3 * (1.0 -retrans['h']['K']) \
                        +s_litterfall[ix] * nut_con['s']['K']*1e-3 * (1.0 -retrans['s']['K'])
        
       #***************** Pine bogs ***************************************
        ix = ix_pine_bog            
        gv_bot[ix] =  np.square(31.809 +0.008*longitude*dem[ix] -3e-4*Nstems*ba[ix] \
                                +6e-5*Nstems*age[ix] -0.188*dem[ix]) -0.5 + 222.22                #Bottom layer total
        #gv_field[ix] =  np.square(48.12 -1e-5*ts**2 +0.013*sfc[ix]*age[ix] -0.04*vol[ix]*age[ix] \
        #                        +0.026*sfc[ix]*vol[ix]) - 0.5 +133.26                                        #Field layer total
        gv_tot[ix] =  np.square(50.098 +0.005*longitude*dem[ix] -1e-5*vol[ix]*Nstems +0.026*sfc[ix]*age[ix] \
                    -1e-3*dem[ix]*ts -0.014*vol[ix]*drain_s) - 0.5 + 167.40                #Total           
        gv_field[ix] = gv_tot[ix] - gv_bot[ix]
              #annual litterfall rates
        ds_litterfall[ix] = fl_share['pine_bog']['ds']*(gv_tot[ix]-gv_bot[ix])*lit_share['ds']*fl_to_total_turnover
        h_litterfall[ix] = fl_share['pine_bog']['h']*(gv_tot[ix]-gv_bot[ix])*lit_share['h']*fl_to_total_turnover
        s_litterfall[ix] = gv_bot[ix]*lit_share['s']
        n_gv[ix] = gv_field[ix] * fl_share['pine_bog']['ds']*nut_con['ds']['N']*1e-3*fl_above_to_total \
                        +gv_field[ix] * fl_share['pine_bog']['h']*nut_con['h']['N']*1e-3*fl_above_to_total \
                        +gv_bot[ix] *nut_con['s']['N']*1e-3
        p_gv[ix] = gv_field[ix] * fl_share['pine_bog']['ds']*nut_con['ds']['P']*1e-3*fl_above_to_total \
                        +gv_field[ix] * fl_share['pine_bog']['h']*nut_con['h']['P']*1e-3*fl_above_to_total \
                        +gv_bot[ix] *nut_con['s']['P']*1e-3
        k_gv[ix] = gv_field[ix] * fl_share['pine_bog']['ds']*nut_con['ds']['K']*1e-3*fl_above_to_total \
                        +gv_field[ix] * fl_share['pine_bog']['h']*nut_con['h']['K']*1e-3*fl_above_to_total \
                        +gv_bot[ix] *nut_con['s']['K']*1e-3
        nup_litter[ix] = ds_litterfall[ix] * nut_con['ds']['N']*1e-3 * (1.0 -retrans['ds']['N']) \
                        +h_litterfall[ix] * nut_con['h']['N']*1e-3 * (1.0 -retrans['h']['N']) \
                        +s_litterfall[ix] * nut_con['s']['N']*1e-3 * (1.0 -retrans['s']['N'])
        pup_litter[ix] = ds_litterfall[ix] * nut_con['ds']['P']*1e-3 * (1.0 -retrans['ds']['P']) \
                        +h_litterfall[ix] * nut_con['h']['P']*1e-3 * (1.0 -retrans['h']['P']) \
                        +s_litterfall[ix] * nut_con['s']['P']*1e-3 * (1.0 -retrans['s']['P'])
        kup_litter[ix] = ds_litterfall[ix] * nut_con['ds']['K']*1e-3 * (1.0 -retrans['ds']['K']) \
                        +h_litterfall[ix] * nut_con['h']['K']*1e-3 * (1.0 -retrans['h']['K']) \
                        +s_litterfall[ix] * nut_con['s']['K']*1e-3 * (1.0 -retrans['s']['K'])

        #**************** Open peatlands**********************************
        # Not in Mäkipää & Muukkonen, apply Pine bogs
        ix = ix_open_peat            
        age[ix] = 10.
        vol[ix] = 5.
        ba[ix] = 1.
        Nstems=100.

        gv_bot[ix] =  np.square(31.809 +0.008*longitude*dem[ix] -3e-4*Nstems*ba[ix] \
                                +6e-5*Nstems*age[ix] -0.188*dem[ix]) -0.5 + 222.22                #Bottom layer total
        #gv_field[ix] =  np.square(48.12 -1e-5*ts**2 +0.013*sfc[ix]*age[ix] -0.04*vol[ix]*age[ix] \
        #                        +0.026*sfc[ix]*vol[ix]) - 0.5 +133.26                                        #Field layer total
        gv_tot[ix] =  np.square(50.098 +0.005*longitude*dem[ix] -1e-5*vol[ix]*Nstems +0.026*sfc[ix]*age[ix] \
                    -1e-3*dem[ix]*ts -0.014*vol[ix]*drain_s) - 0.5 + 167.40                #Total           
        gv_field[ix] = gv_tot[ix] - gv_bot[ix]
              #annual litterfall rates
        ds_litterfall[ix] = fl_share['pine_bog']['ds']*(gv_tot[ix]-gv_bot[ix])*lit_share['ds']*fl_to_total_turnover
        h_litterfall[ix] = fl_share['pine_bog']['h']*(gv_tot[ix]-gv_bot[ix])*lit_share['h']*fl_to_total_turnover
        s_litterfall[ix] = gv_bot[ix]*lit_share['s']
        n_gv[ix] = gv_field[ix] * fl_share['pine_bog']['ds']*nut_con['ds']['N']*1e-3*fl_above_to_total \
                        +gv_field[ix] * fl_share['pine_bog']['h']*nut_con['h']['N']*1e-3*fl_above_to_total \
                        +gv_bot[ix] *nut_con['s']['N']*1e-3
        p_gv[ix] = gv_field[ix] * fl_share['pine_bog']['ds']*nut_con['ds']['P']*1e-3*fl_above_to_total \
                        +gv_field[ix] * fl_share['pine_bog']['h']*nut_con['h']['P']*1e-3*fl_above_to_total \
                        +gv_bot[ix] *nut_con['s']['P']*1e-3
        k_gv[ix] = gv_field[ix] * fl_share['pine_bog']['ds']*nut_con['ds']['K']*1e-3*fl_above_to_total \
                        +gv_field[ix] * fl_share['pine_bog']['h']*nut_con['h']['K']*1e-3*fl_above_to_total \
                        +gv_bot[ix] *nut_con['s']['K']*1e-3
        nup_litter[ix] = ds_litterfall[ix] * nut_con['ds']['N']*1e-3 * (1.0 -retrans['ds']['N']) \
                        +h_litterfall[ix] * nut_con['h']['N']*1e-3 * (1.0 -retrans['h']['N']) \
                        +s_litterfall[ix] * nut_con['s']['N']*1e-3 * (1.0 -retrans['s']['N'])
        pup_litter[ix] = ds_litterfall[ix] * nut_con['ds']['P']*1e-3 * (1.0 -retrans['ds']['P']) \
                        +h_litterfall[ix] * nut_con['h']['P']*1e-3 * (1.0 -retrans['h']['P']) \
                        +s_litterfall[ix] * nut_con['s']['P']*1e-3 * (1.0 -retrans['s']['P'])
        kup_litter[ix] = ds_litterfall[ix] * nut_con['ds']['K']*1e-3 * (1.0 -retrans['ds']['K']) \
                        +h_litterfall[ix] * nut_con['h']['K']*1e-3 * (1.0 -retrans['h']['K']) \
                        +s_litterfall[ix] * nut_con['s']['K']*1e-3 * (1.0 -retrans['s']['K'])
        
        
        
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
        gv_bot[ix_cc] = gv_bot[ix_cc] * to_cc

        return n_gv, p_gv, k_gv, nup_litter, pup_litter, kup_litter, gv_bot
        

    # initial N and P mass, kg ha-1    
    n_gv, p_gv, k_gv, nup_litter, pup_litter, kup_litter, gv_tot = gv_biomass_and_nutrients(n,  ix_spruce_mire, ix_pine_bog,
                ix_open_peat, latitude, longitude, dem, ts, sfc, vol, Nstems, ba, drain_s, age)
 
    # ground vegetation mass at the end of simulation, kg ha-1    
    vol = vol + expected_yield
    age = age + simtime
    n_gv_end, p_gv_end, k_gv_end, nup_litter_end, pup_litter_end, kup_litter_end, gv_tot = gv_biomass_and_nutrients(n, ix_spruce_mire, ix_pine_bog,
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
    
    
    
    return nup, pup, kup


base = 5.0    
dd = forc['T']-base    
dd[dd<base] = 0.0
dds = dd.sum(axis=0)                                              #cumulative temperature sum degree days    
yrs = np.shape(forc)[0] / 365.25
ts = (dds/yrs)
simtime = yrs

mottipath = r'C:/Users/alauren/Documents/WinPython-64bit-2.7.10.3/Susi_7_0_py27/motti2/'
mottifile = mottipath + 'Viitasaari_Mtkg.xls'
length = np.shape(forc)[0]
agearray = ageSim + np.arange(0,length,1.)/365.

#return ageToHdom(a_arr), ageToLAI(a_arr), ageToVol(a_arr), ageToYield(a_arr), ageToBm(a_arr), ageToBa, \
#        bmToLeafMass, bmToHdom, bmToYi, yiToBm, ageToVol, bmToLitter, bmToStems, volToLogs, volToPulp

hdom, LAI, vol, yi, bm, ba, bmToLeafMass, bmToHdom, bmToYi, yiToBm,  \
    ageToVol, bmToLitter, bmToStems, volToLogs, volToPulp = motti_development(spara, agearray, mottifile)             # dom hright m, LAI m2m-2, vol m3/ha, yield m3/ha, biomass kg/ha

_, sp = get_motti(mottifile, return_spe=True)
sp =sp[0]

n =20
lat, lon = forc['lat'][0], forc['lon'][0]
stems = (1200,1000)
ba = (ba[0], ba[-1])
stems = (stems[0], stems[-1])
yi = (yi[0], yi[-1])
understory_uptake(n, lat, lon, ba , stems, yi, sp, ts, simtime, spara['sfc'])