# -*- coding: utf-8 -*-
"""
Created on Mon May 21 18:38:10 2018

@author: lauren
"""
import numpy as np
import pandas as pd
import datetime
import matplotlib.pylab as plt
import seaborn as sns
import time

from canopygrid import CanopyGrid
from mosslayer import MossLayer
from strip import StripHydrology, drain_depth_development
from temperature import PeatTemperature
from docclass import DocModel

import susi_io
from susi_utils import read_FMI_weather
from susi_utils import heterotrophic_respiration_yr, CH4_flux_yr
from susi_utils import nutrient_release,  rew_drylimit, nutrient_demand, nut_to_vol
from susi_utils import motti_development,  get_motti, assimilation_yr
from susi_utils import get_mese_input, get_mese_out, understory_uptake, get_temp_sum
from susi_para import get_susi_para

def run_susi(forc, wpara, cpara, org_para, spara, outpara, photopara, start_yr, end_yr, wlocation=None, mottifile=None, peat=None, 
             photosite=None, folderName=None, hdomSim=None, volSim=None, ageSim=None, 
             sarkaSim=None, sfc=None, susiPath = None, simLAI=None, kaista=None, sitename=None): 
    
    print ('******** Susi-peatland simulator v.9 (2021) c Ari Laurén *********************')
    print ('           ')    
    print ('Initializing stand and site:') 
     
    dtc = cpara['dt']                                                         # canopy model timestep

    start_date = datetime.datetime(start_yr,1,1); end_date=datetime.datetime(end_yr,12,31)
    length = (end_date - start_date).days +1
    yrs = end_yr - start_yr +1
    ts = get_temp_sum(forc)                                                    # temperature sum
        
    lat=forc['lat'][0]; lon=forc['lon'][0]                                     # location of weather file, determines the simulation location
    print ('      - Weather input:', wpara['description'], ', start:', start_yr, ', end:', end_yr) 
    print ('      - Latitude:', lat, ', Longitude:', lon )
    susi_io.print_site_description(spara)                                               # Describe site parameters for user
    agearray = ageSim + np.arange(0,length,1.)/365.
    
    hdom, LAI, vol, yi, bm, ba, \
        stems, bmToLeafMass, bmToLAI, bmToHdom, bmToYi, bmToBa, yiToVol, yiToBm, \
        ageToVol, bmToLitter, bmToStems, volToLogs, volToPulp, sp,  \
        N_demand, P_demand, K_demand = motti_development(spara, agearray, mottifile)             # dom hright m, LAI m2m-2, vol m3/ha, yield m3/ha, biomass kg/ha
    
    
    if outpara['static stand']: 
        print ('     // Working with static stand, hdom' ,spara['hdom'], 'LAI', LAI[0], 'vol', spara['vol']) 
        hdom = np.ones(length)*spara['hdom']
        LAI = np.ones(length)*simLAI
        vol = np.ones(length)*spara['vol']
    else:
        spara['vol']= vol[0]                                                    #these are for printing purposes only
        spara['hdom']=hdom[0]
    
    #********* Above ground hydrology initialization ***************
    cmask = np.ones(spara['n'])                                                # compute canopy and moss for each soil column (0, and n-1 are ditches??)
    cstate = cpara['state'].copy()
    for key in cstate.keys():
        cstate[key] *= cmask
    cpy = CanopyGrid(cpara, cstate, outputs=False)                             # initialize above ground hydrology model

    for key in org_para.keys():                                                 
        org_para[key] *= cmask
    moss = MossLayer(org_para, outputs=True)                                   # initialize moss layer hydrologu
    print ('Canopy and moss layer hydrology initialized')

    #******** Soil and strip parameterization *************************
    stp = StripHydrology(spara)                                                # initialize soil hycrology model
    pt = PeatTemperature(spara, forc['T'].mean())                              # initialize peat temperature model       
    n = spara['n']        
    docs = DocModel(spara['nLyrs'], n, spara['bd top'], spara['bd bottom'], spara['dzLyr'], length)   # initialize DOC model
    print ('Soil hydrology, temperature and DOC models initialized')

    """
    change these to nodewise deltas, ets
    """                                                                        # number of computation nodes
    deltas = np.zeros((length, n))                                                  # Infliltration-evapotranspiration, mm/day    
    ets = np.zeros((length, n))                                                     # Evapotranspiration, mm/day
    dt= 1.                                                                     # time step, days
    summer_dwt=[];  co2_respi=[]                                               # output variables for figures
    
    #initialize result arrays
    scen=spara['scenario name']; rounds= len(spara['ditch depth east'])
    dwts = np.zeros((rounds, int(length/dt),n), dtype=float)                   # water table depths, m,  ndarray(scenarios, days, number of nodes)
    afps = np.zeros((rounds, int(length/dt),n), dtype=float)                   # air-filled porosity (m3 m-3),  ndarray(scenarios, days, number of nodes)
    hts = np.zeros((rounds, int(length/dt),n), dtype=float)                    # water table depths, m,  ndarray(scenarios, days, number of nodes)
    air_ratios =np.zeros((rounds, int(length/dt),n), dtype=float)              # ratio of afp in root zone to total
    co2release = np.zeros((rounds, int(length/dt)), dtype=float)
    peat_temperatures = np.zeros((rounds, int(length/dt), spara['nLyrs']))     # daily peat temperature profiles

    c_bals_yr = np.zeros((rounds, yrs,n), dtype=float)                         # annual spoil/peat C balance in nodes kg C ha-1
    c_balstrees_yr = np.zeros((rounds, yrs,n), dtype=float)                    # annual stand C balance in nodes kg C ha-1
    n_export_yr = np.zeros((rounds, yrs,n), dtype=float)                       # annual leaching of N to water course kg ha-1
    p_export_yr= np.zeros((rounds, yrs,n), dtype=float)
    k_export_yr= np.zeros((rounds, yrs,n), dtype=float)
    CH4_yr = np.zeros((rounds, yrs,n), dtype=float)
    
    npps = np.zeros((rounds, yrs,n), dtype=float)                              # net primary production kg biomass ha-1 yr-1    
    het = np.zeros((rounds, yrs,n), dtype=float)                               # heterotrophic respiration kg CO2 ha-1 yr-1
    growths = np.zeros((rounds,yrs,n), dtype=float)                            # no used
    g_npps = np.zeros((rounds,yrs,n), dtype=float)                             # stand volume allowed by npp and physical restrictions m3 ha-1
    g_npps_pot = np.zeros((rounds,yrs,n), dtype=float)                         # potential stand volume allowed by npp alone m3 ha-1 
    g_nuts = np.zeros((rounds,yrs,n), dtype=float)                             # stand volume allowed by supply of growth-limiting nutrient m3 ha-1
    g_Ns = np.zeros((rounds,yrs,n), dtype=float)                               # stand volume allowed by supply of N m3 ha-1
    g_Ps = np.zeros((rounds,yrs,n), dtype=float)                               # stand volume allowed by supply of P m3 ha-1
    g_Ks = np.zeros((rounds,yrs,n), dtype=float)                               # stand volume allowed by supply of K m3 ha-1
    yis = np.zeros((rounds,yrs,n), dtype=float)                                # not used
    vols = np.zeros((rounds,yrs,n), dtype=float)                               # realized stand volumes m3 ha-1

    end_vols = np.zeros((rounds, n), dtype=float)    
    c_bals = np.zeros((rounds, n), dtype=float)
    c_bals_trees = np.zeros((rounds, n), dtype=float)
    sdwt = np.zeros((rounds, n), dtype=float)
    CH4release = np.zeros((rounds, n), dtype=float)
    Nrelease = np.zeros((rounds, n), dtype=float)    
    Prelease = np.zeros((rounds, n), dtype=float)
    Krelease = np.zeros((rounds, n), dtype=float)

    biomass_gr = np.zeros((rounds, n))
    litterfall_gv_cumul = np.zeros((rounds, n))
    runoff =np.zeros((rounds, int(length/dt)), dtype=float)
    swes = np.zeros((rounds, int(length/dt)), dtype=float)
    
    Nstorage = np.zeros(n, dtype=float)                                        # Nodewise nutrient storage in the rooting zone kg/ha
    Pstorage = np.zeros(n, dtype=float)                                        # Nodewise nutrient storage in the rooting zone kg/ha
    Kstorage = np.zeros(n, dtype=float)                                        # Nodewise nutrient storage in the rooting zone kg/ha
    
    Nout = np.zeros((rounds,n))                                                # Nodewise potential for N export kg/ha
    Pout = np.zeros((rounds,n))                                                # Nodewise potential for P export kg/ha
    Kout = np.zeros((rounds,n))                                                # Nodewise potential for K export kg/ha
    HMWDOCout = np.zeros((rounds, n))                                          # Nodewise potential for HMWDOC export kg/ha
    LMWDOCout = np.zeros((rounds, n))                                          # Nodewise potential for LMWDOC export kg/ha
    
    for r, dr in enumerate(zip(spara['ditch depth west'], spara['ditch depth 20y west'], spara['ditch depth east'], spara['ditch depth 20y east'])):   #SCENARIO loop

        dwt=spara['initial h']*np.ones(spara['n'])           
        hdr_west, hdr20y_west,hdr_east, hdr20y_east = dr                                                        # drain depth [m] in the beginning and after 20 yrs
        h0ts_west = drain_depth_development(length, hdr_west, hdr20y_west)                     # compute daily values for drain bottom boundary condition
        h0ts_east = drain_depth_development(length, hdr_east, hdr20y_east)                     # compute daily values for drain bottom boundary condition

        v, leaf_mass, hc, b = vol[0]*np.ones(n), bmToLeafMass(bm[0])*np.ones(n), hdom[0]*np.ones(n), bm[0]*np.ones(n)
        b_ini = yiToBm(v) #b.copy()
        b = b_ini.copy()
        BA0=bmToBa(b)
        yi0 = v.copy()
        stems0 = bmToStems(b)
        ageyrs = ageSim + np.array(range(yrs))+1

        # ---- Initialize integrative output arrays (outputs in nodewise sums) -------------------------------
        start = 0            
        litter_cumul = np.zeros(n)
        Crelease = np.zeros(n)
        Nleach = np.zeros(n)
        Pleach = np.zeros(n)
        Kleach = np.zeros(n)
        DOCleach = np.zeros(n)
        HMWleach = np.zeros(n)
        
        bm_deadtrees = np.zeros(n)
        stems = bmToStems(b_ini)                                                #current number of stems in the stand  (ha-1)
        n_deadtrees = np.maximum((stems - bmToStems(b)), np.zeros(n))


        print ('***********************************')        
        print ('Computing canopy and soil hydrology ', length, ' days', 'scenario:', scen[r])
        stp.reset_domain()   
        pt.reset_domain()
        docs.reset_domain()
        d = 0                                                                  # day index
        start = 0
        year = 0
        for yr in range(start_yr, end_yr+1):                                   # year loop 
            days = (datetime.datetime(yr,12, 31) - datetime.datetime(yr,1, 1)).days+1
            hc = bmToHdom(b)   
            lai =  bmToLAI(b) 
            for dd in range(days):                                             # day loop   
                #-------Canopy hydrology--------------------------            
                reww = rew_drylimit(dwt)                                       # for each column: moisture limitation from ground water level (Feddes-function)            
                doy = forc.iloc[d, 14]
                ta =  forc.iloc[d, 4]
                vpd = forc.iloc[d, 13]
                rg = forc.iloc[d, 8]
                par = forc.iloc[d, 10]
                prec=forc.iloc[d, 7]/86400.
    
                potinf, trfall, interc, evap, ET, transpi, efloor, MBE, SWE = cpy.run_timestep(doy, dtc, ta, prec, rg, par, vpd, 
                                                                hc=hc, LAIconif=lai, Rew=reww, beta=moss.Ree) # kaikki (käytä tätä)
                potinf, efloor, MBE2 = moss.interception(potinf, efloor)
                deltas[d] = potinf - transpi
                ets[d] = efloor + transpi       
    
                if d%365==0: print ('  - day #', d, ' hdom ', np.round(np.mean(hc),2), ' m, ',  
                                    'LAI ', np.round(np.mean(lai),2), ' m2 m-2')
    
                #--------Soil hydrology-----------------
                dwt, ht, roff, air_ratio, afp = stp.run_timestep(d, h0ts_west[d], h0ts_east[d], deltas[d,:], moss)
                dwts[r,d,:] = dwt
                hts[r,d,:] = ht            
                air_ratios[r,d,:]= air_ratio
                afps[r,d,:] = afp
                runoff[r,d] = roff
                swes[r,d] = np.mean(SWE)
                z, peat_temperature = pt.run_timestep(ta, np.mean(SWE), np.mean(efloor))
                peat_temperatures[r,d,:] = peat_temperature
                d += 1
            
        #----- Hydrology and temperature-related variables to time-indexed dataframes -----------------
            sday = datetime.datetime(yr, 1, 1)
            t5 = pd.DataFrame(peat_temperatures[r,start:start+days,1],index=pd.date_range(sday,periods=days))   #Peat temperature in 5 cm depth, deg C        
            df_peat_temperatures = pd.DataFrame(peat_temperatures[r,start:start+days,:],index=pd.date_range(sday,periods=days))
            dfwt = pd.DataFrame(dwts[r,start:start+days,:],index=pd.date_range(sday,periods=days))
            dfair_r = pd.DataFrame(air_ratios[r,start:start+days,:], index = pd.date_range(sday,periods=days))
            dfafp = pd.DataFrame(afps[r,start:start+days,:],index=pd.date_range(sday,periods=days))
                   
        # ----------  Computing biogeochemistry -----------------------
            
            nup_gv, pup_gv, kup_gv, litterfall_gv, gv_leafmass = understory_uptake(spara['n'], lat, lon, 
                                                            BA0, bmToBa(b), stems0, bmToStems(b), yi0, 
                                                            bmToYi(b), sp, ts, 1, spara['sfc'], ageSim+year)
            litterfall_gv_cumul[r,:] = litterfall_gv_cumul[r,:] + litterfall_gv
            _, co2, Rhet, Rhet_root = heterotrophic_respiration_yr(t5, yr, dfwt, dfair_r, v, spara) #Rhet in kg/ha/yr CO2            
            days = len(co2) 
            CH4, CH4mean, CH4asCO2eq = CH4_flux_yr(yr, dfwt)                   # annual ch4 nodewise (kg ha-1 yr-1), mean ch4, and mean ch4 as co2 equivalent
            co2release[r,start:start+days] = co2                               # mean daily time series for co2 efflux kg/ ha/day CO2
            Ns,Ps,Ks = nutrient_release(spara['sfc'],spara['sfc_specification'], Rhet_root, N=spara['peatN'], P=spara['peatP'], K=spara['peatK']) # N P K release in kg/ha/yr                              #supply of N,P,K kg/ha/timestep
            Nstot, Pstot, Kstot = nutrient_release(spara['sfc'], spara['sfc_specification'],Rhet, N=spara['peatN'], P=spara['peatP'], K=spara['peatK'])
            
            Nleach = Nleach + Nstot - Ns
            Pleach = Pleach + Pstot - Ps
            Kleach = Kleach + Kstot - Ks
            n_export_yr[r,year,:] = Nstot - Ns
            p_export_yr[r,year,:] = Pstot - Ps
            k_export_yr[r,year,:] = Kstot - Ks
            CH4_yr[r,year,:] = CH4
            
            doc, hmw = docs.doc_release(df_peat_temperatures.loc[str(yr)], dfwt.loc[str(yr)])            
            DOCleach = DOCleach + doc
            HMWleach = HMWleach + hmw

            #------------------fertilization--------------------------------
            fert={'N':0.0, 'P':0.0, 'K':0.0}
            if yr >= spara['fertilization']['application year']:
                tfert = yr-spara['fertilization']['application year']
                for nutr in ['N', 'P', 'K']:
                    nut_efficiency = spara['fertilization'][nutr]['eff']
                    dose = spara['fertilization'][nutr]['dose']
                    decay_k =spara['fertilization'][nutr]['decay_k']
                    fert[nutr] = (dose*np.exp(-decay_k*tfert) - dose*np.exp(-decay_k*(tfert+1)))*nut_efficiency

            #----------------------------------------------------------------
            Ns, Ps, Ks = Ns+spara['depoN']+fert['N'], Ps+spara['depoP']+fert['P'], Ks+spara['depoK']+fert['K']        #decomposition + deposition from Ruoho-Airola et al 2003 Fig.4
             
            Nrelease[r,:] = Nrelease[r,:] + Ns 
            Prelease[r,:] = Prelease[r,:] + Ps 
            Krelease[r,:] = Krelease[r,:] + Ks 
            Crelease = Crelease + Rhet*(12./44)                                # CO2 to C, annual sum, nodewise in kg C ha-1
            CH4release[r,:] = CH4release[r,:]  + CH4                                     # Total nodewise CH4 release in the simulation, kg CH4 ha-1 
            NPP, NPP_pot = assimilation_yr(photopara, forc.loc[str(yr)], dfwt.loc[str(yr)], dfafp.loc[str(yr)], leaf_mass, hc, species = spara['species'])     # NPP nodewise, kg organic matter /ha /yr sum over the year

            bm_change =  NPP - n_deadtrees/stems * b - bmToLitter(b)*365.
            bm_change_pot =  NPP_pot - n_deadtrees/stems * b - bmToLitter(b)*365.
            new_bm = b + np.maximum(bm_change, np.zeros(n))                    # suggestion for new biomass kg/ha
            new_bm_pot = b + np.maximum(bm_change_pot, np.zeros(n))
            g_npp = bmToYi(new_bm)                                               # suggested bm to new volume as yield m3/ha
            g_npp_pot = bmToYi(new_bm_pot)

            g_npps[r,yr-start_yr,:] = g_npp
            g_npps_pot[r,yr-start_yr,:] = g_npp_pot
            g_N, g_P, g_K = nut_to_vol(v, Ns,Ps,Ks,bmToLitter(b)*365., nup_gv, pup_gv, 
                                       kup_gv, leaf_mass*1000, gv_leafmass )              # volume growth allowed by nutrient release litter here in kg/ha/yr
            g_Ns[r,yr-start_yr,:] = g_N
            g_Ps[r,yr-start_yr,:] = g_P
            g_Ks[r,yr-start_yr,:] = g_K
            
            
            lim_nut_gr = np.minimum(g_K, g_P)                                  # find the growth limiting factor
            lim_nut_gr = np.minimum(lim_nut_gr, g_N)       
            #g_nut = v + lim_nut_gr
            g_nuts[r,yr-start_yr,:] = lim_nut_gr               
            v = np.minimum(lim_nut_gr, g_npp)                                  # new volume as yield 
            
            BA0 = bmToBa(b) 
            stems0 = bmToStems(b)
            yi0 = bmToYi(b)            #update old: basal area, stem number, volume
            vols[r,yr-start_yr,:] = v 
            
            bm_restr = yiToBm(v)                                                
            
            #------Annual balances--------------------
            c_bals_yr[r, yr-start_yr, :] = (bmToLitter(b)*365. + n_deadtrees/stems * b + litterfall_gv) * 0.5 - Rhet*(12./44)    #(rounds, yrs,n)
            c_balstrees_yr[r, yr-start_yr, :] = ((bm_restr-b) + bmToLitter(b)*365. + n_deadtrees/stems * b + litterfall_gv) * 0.5 - Rhet*(12./44)      # (rounds, yrs,n)
           
            litter_cumul = litter_cumul + bmToLitter(b)*365   
            
            leaf_mass, hc, b = bmToLeafMass(bm_restr), bmToHdom(bm_restr), bm_restr   
            n_deadtrees = np.maximum((stems - bmToStems(b)), np.zeros(n))
            bm_deadtrees = bm_deadtrees + n_deadtrees/stems * b
            stems = bmToStems(b)
            start = start+days    
            year +=1


    # ------------------ End biogeochemistry loop, end yr loop-------------------------------------
            
        
        # Carbon balance of the stand in kg / ha / simulation time
        #docs.draw_doc(sitename, yrs)
        end_vols[r, :] = v
        c_bals_trees[r, :] = ((b-b_ini) + litter_cumul + bm_deadtrees + litterfall_gv_cumul[r,:]) * 0.5 - Crelease     # in kg C /ha/time, 600 is the contribution of ground vegetation (Minkkinen et al. 20018)
        c_bals[r, :] = (litter_cumul + bm_deadtrees + litterfall_gv_cumul[r,:]) * 0.5 - Crelease     # in kg C /ha/time, 600 is the contribution of ground vegetation (Minkkinen et al. 20018)
        biomass_gr[r, :] = (b-b_ini) + litter_cumul + bm_deadtrees
        Nout[r, :] = Nleach
        Pout[r, :] = Pleach
        Kout[r, :] = Kleach
        HMWDOCout[r, :] =  HMWleach 
        LMWDOCout[r, :] = DOCleach - HMWleach 
             

        v_start = vol[0]*np.ones(n)
        potential_gr = []; phys_restr=[]; chem_restr=[]; N_gr = []; P_gr=[]; K_gr=[]; phys_gr=[]
        for yr in range(start_yr, end_yr+1):   
            vgrowth = vols[0,yr-start_yr, :]-v_start
            pot_growth = g_npps_pot[0, yr-start_yr, :]-v_start
            phys_growth = g_npps[0, yr-start_yr, :]-v_start
            chem_growth = g_nuts[0, yr-start_yr, :]-v_start
            n_gr = g_Ns[0, yr-start_yr, :]-v_start
            p_gr = g_Ps[0, yr-start_yr, :]-v_start
            k_gr = g_Ks[0, yr-start_yr, :]-v_start
            
            v_start=vols[0,yr-start_yr, :]
            
            potential_gr.append(np.mean(pot_growth))
            phys_gr.append(np.mean(phys_growth))
            phys_restr.append(np.mean(phys_growth)/np.mean(pot_growth))
            chem_restr.append(np.mean(chem_growth)/np.mean(pot_growth))
            N_gr.append(n_gr)
            P_gr.append(p_gr)
            K_gr.append(k_gr)

        print ('***********Growth restrictions*********************')
        print (np.mean(np.array(potential_gr)), np.mean(np.array(phys_restr)),np.mean(np.array(chem_restr)))
        print ('grs', np.mean(np.array(potential_gr)), np.mean(np.array(vgrowth)), \
               np.mean(np.array(N_gr)), np.mean(np.array(P_gr)), np.mean(np.array(K_gr)), np.mean(np.array(phys_gr)))
        #if outpara['figs']:
        #    susi_io.fig_stand_growth_node(r, rounds, ageSim, start_yr, end_yr, ageToVol, 
        #            ageyrs,g_npps[:,:, 1:-1], g_nuts[:,:, 1:-1], spara['scenario name'][r])    
        
        
        #daily_dwt = [np.mean(dwts[r,d,1:-1]) for d in range(length)] 
 
        # ******** Outputs *************************************
        
        # Change delatas ets to np.mean(deltas), np.mean(ets)
        susi_io.print_scenario_nodes(r,c_bals_trees, np.mean(deltas,axis=1), np.mean(ets, axis=1), h0ts_west, h0ts_east, dwts, bmToYi, g_nuts, end_vols, yi )         
        co2_respi.append(np.mean(np.sum(het[r,:,1:-1], axis=0)))
            

        summer_mean_dwt, summer = susi_io.output_dwt_growing_season(dwts[r,:], length, start_yr, end_yr, start_date, outpara, wpara, scen[r])
        sdwt[r,:] = summer_mean_dwt
        

        if outpara['to_file']: 
            susi_io.dwt_to_excel(dwts[r,:,:],outpara, scen[r])
            susi_io.runoff_to_excel(runoff[r,:]*1000., swes[r,:], outpara, scen[r])
            #susi_io.write_excel(wlocation, wpara, spara, outpara, LAI, hdom, h0ts_west[0], h0ts_east[0],summer, summer_median_dwt)
            susi_io.c_and_nut_to_excel(c_bals_yr[r,:,:], c_balstrees_yr[r,:,:], 
                                       n_export_yr[r,:,:], p_export_yr[r,:,:],
                                       k_export_yr[r,:,:], outpara, scen[r])

        # Change delatas ets to np.mean(deltas), np.mean(ets)        
        if outpara['hydfig']: susi_io.fig_hydro(stp.ele, hts[r,:,:], spara, wpara, wlocation, np.mean(ets*1000., axis=1), forc['Prec'].values, \
                            forc['T'].values, co2release[r,:], LAI, hdr_west, hdr_east, runoff[r,:], scen[r])

    # ******************* End ditch depth scenario loop*********************
    
    
    #if outpara['figs']: susi_io.outfig(summer_dwt, co2_respi, growths-growths[0],spara['ditch depth'], relative_response, rounds)
    if outpara['figs']:
        susi_io.fig_stand_growth_node(rounds, ageSim, start_yr, end_yr, ageToVol, 
                    ageyrs,vols[:,:, 1:-1], spara['scenario name'], dwts)
    
    #These are for MESE-simulations only until return
    opt_strip=True   #True: normal simulation, False: mese-simulations
    if opt_strip or kaista is None:
        fr = 1 ; to = -1; frw = 1; tow  = -1        

    #Change yield volumes to standing volumes
    
    end_vols = yiToVol(end_vols) 
    
    gr = np.array([end_vols[k]-end_vols[0] for k in range(rounds)])
    gr = np.array([np.mean(gr[k][fr:to]) for k in range(rounds)])        
    cbt = np.array([c_bals_trees[k] - 0.0 for k in range(rounds)])
    cbt = np.array([np.mean(cbt[k][fr:to])/yrs for k in range(rounds)]) 
    dcbt = np.array([c_bals_trees[k]- c_bals_trees[0] for k in range(rounds)])
    dcbt = np.array([np.mean(dcbt[k][fr:to])/yrs for k in range(rounds)]) 
    cb = np.array([c_bals[k] - 0.0 for k in range(rounds)])
    cb = np.array([np.mean(cb[k][fr:to]) for k in range(rounds)]) 
    dcb = np.array([c_bals[k]- c_bals[0] for k in range(rounds)])
    dcb = np.array([np.mean(dcb[k][fr:to]) for k in range(rounds)]) 
    w =  np.array([sdwt[k] - 0.0 for k in range(rounds)])
    w = np.array([np.mean(w[k][fr:to]) for k in range(rounds)])                
    dw =  np.array([sdwt[k]-sdwt[0] for k in range(rounds)])
    dw = np.array([np.mean(dw[k][fr:to]) for k in range(rounds)])        
    v_end = np.array([end_vols[k] - 0.0 for k in range(rounds)])
    v_end = np.array([np.mean(v_end[k][fr:to]) for k in range(rounds)])
    logs = np.array(volToLogs(v_end))
    pulp = np.array(volToPulp(v_end))        
    dv = np.array([v_end[k]-v_end[0] for k in range(rounds)])
    dlogs = np.array([logs[k]-logs[0] for k in range(rounds)])
    dpulp = np.array([pulp[k]-pulp[0] for k in range(rounds)])
    bms = np.array([biomass_gr[k] for k in range(rounds)])
    bms = np.array([np.mean(bms[k][fr:to]) for k in range(rounds)])
    annual_runoff = np.sum(runoff, axis = 1)*1000. /yrs
    drunoff = [annual_runoff[k] - annual_runoff[0] for k in range(rounds)]
    nout = [np.mean(Nout[k])/yrs for k in range(rounds)]
    pout = [np.mean(Pout[k])/yrs for k in range(rounds)]
    kout = [np.mean(Kout[k])/yrs for k in range(rounds)]
    hmwdocout = [np.mean(HMWDOCout[k])/yrs for k in range(rounds)]
    lmwdocout = [np.mean(LMWDOCout[k])/yrs for k in range(rounds)]
    
    nrelease = [np.mean(Nrelease[k])/yrs for k in range(rounds)]
    prelease = [np.mean(Prelease[k])/yrs for k in range(rounds)]
    krelease = [np.mean(Krelease[k])/yrs for k in range(rounds)]
    ch4release = [np.mean(CH4release[k])/yrs for k in range(rounds)]
    
    
    return vol[0], v_end, (v_end-vol[0])/yrs, cbt, dcbt, cb, dcb,  w, dw, logs, pulp, dv, dlogs, dpulp, yrs, bms/yrs, \
                    nout, pout, kout, hmwdocout, annual_runoff, nrelease, prelease, krelease, ch4release 

          