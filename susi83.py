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

import susi_io
from susi_utils import read_FMI_weather
from susi_utils import heterotrophic_respiration_yr
from susi_utils import nutrient_release,  rew_drylimit, nutrient_demand, nut_to_vol
from susi_utils import motti_development,  get_motti, assimilation_yr
from susi_utils import get_mese_input, get_mese_out, understory_uptake, get_temp_sum
from susi_para import get_susi_para

def run_susi(forc, wpara, cpara, org_para, spara, outpara, photopara, start_yr, end_yr, wlocation=None, mottifile=None, peat=None, 
             photosite=None, folderName=None, hdomSim=None, volSim=None, ageSim=None, 
             sarkaSim=None, sfc=None, susiPath = None, simLAI=None, kaista=None): 
    
    print ('******** Susi-peatland simulator v.8.0 (2020) c Ari Laurén *********************')
    print ('           ')    
    print ('Initializing stand and site:') 
     
    dtc = cpara['dt']                                                         # canopy model timestep

    start_date = datetime.datetime(start_yr,1,1); end_date=datetime.datetime(end_yr,12,31)
    length = (end_date - start_date).days +1
    yrs = end_yr - start_yr +1

    dates = [start_date + datetime.timedelta(days=n) for n in range(length)]         # dates in simulation
        
    lat=forc['lat'][0]; lon=forc['lon'][0]                                     # location of weather file, determines the simulation location
    print ('      - Weather input:', wpara['description'], ', start:', start_yr, ', end:', end_yr) 
    print ('      - Latitude:', lat, ', Longitude:', lon )
    susi_io.print_site_description(spara)                                               # Describe site parameters for user
    agearray = ageSim + np.arange(0,length,1.)/365.
   
    
    
    hdom, LAI, vol, yi, bm, ba, stems, bmToLeafMass, bmToHdom, bmToYi, yiToVol, yiToBm, \
        ageToVol, bmToLitter, bmToStems, volToLogs, volToPulp, sp,  \
        N_demand, P_demand, K_demand = motti_development(spara, agearray, mottifile)             # dom hright m, LAI m2m-2, vol m3/ha, yield m3/ha, biomass kg/ha
    
    
    # here understorey uptake; n, p, k
    ts = get_temp_sum(forc)
    nup_gv, pup_gv, kup_gv, litterfall_gv, gv_leafmass = understory_uptake(spara['n'], lat, lon, ba , stems, yi, sp, ts, yrs, spara['sfc'], ageSim)
    nup_gv = np.minimum(nup_gv, np.ones(spara['n'])*N_demand)
    pup_gv= np.minimum(pup_gv, np.ones(spara['n'])*P_demand)
    kup_gv= np.minimum(kup_gv, np.ones(spara['n'])*K_demand)
    
    
    if outpara['static stand']: 
        print ('     // Working with static stand, hdom' ,spara['hdom'], 'LAI', LAI[0], 'vol', spara['vol']) 
        hdom = np.ones(length)*spara['hdom']
        LAI = np.ones(length)*simLAI
        vol = np.ones(length)*spara['vol']
    else:
        spara['vol']= vol[0]                                                    #these are for printing purposes only
        spara['hdom']=hdom[0]
    
    #********* Above ground hydrology computing***************
    cmask = np.ones(spara['n'])  # compute canopy and moss for each soil column (0, and n-1 are ditches??)
    cstate = cpara['state'].copy()
    for key in cstate.keys():
        cstate[key] *= cmask
    cpy = CanopyGrid(cpara, cstate, outputs=False)
    print ('Canopy initialized')

    # initialize moss layer for each soil column
    for key in org_para.keys():
        org_para[key] *= cmask
    moss = MossLayer(org_para, outputs=True)

    #******** Soil and strip parameterization *************************
    stp = StripHydrology(spara)
    pt = PeatTemperature(spara, forc['T'].mean())
    n= spara['n']        

    """
    change these to nodewise deltas, ets
    """                                                                        # number of computation nodes
    deltas = np.zeros(length)                                                  # Infliltration-evapotranspiration, mm/day    
    ets = np.zeros(length)                                                     # Evapotranspiration, mm/day
    dt= 1.                                                                     # time step, days
    summer_dwt=[]; growth_response=[]; co2_respi=[]                            #output variables for figures
    
    #initialize result arrays
    scen=spara['scenario name']; rounds= len(spara['ditch depth east'])
    dwts = np.zeros((rounds, int(length/dt),n), dtype=float)                    # water table depths, m,  ndarray(scenarios, days, number of nodes)
    afps = np.zeros((rounds, int(length/dt),n), dtype=float)                   # air-filled porosity (m3 m-3),  ndarray(scenarios, days, number of nodes)
    hts = np.zeros((rounds, int(length/dt),n), dtype=float)                    # water table depths, m,  ndarray(scenarios, days, number of nodes)
    air_ratios =np.zeros((rounds, int(length/dt),n), dtype=float)               # ratio of afp in root zone to total
    co2release = np.zeros((rounds, int(length/dt)), dtype=float)
    peat_temperatures = np.zeros((rounds, int(length/dt), spara['nLyrs']))      #daily peat temperature profiles

    npps = np.zeros((rounds, yrs,n), dtype=float)    
    het = np.zeros((rounds, yrs,n), dtype=float)
    growths = np.zeros((rounds,yrs,n), dtype=float)
    g_npps = np.zeros((rounds,yrs,n), dtype=float)
    g_npps_pot = np.zeros((rounds,yrs,n), dtype=float)
    g_nuts = np.zeros((rounds,yrs,n), dtype=float)
    yis = np.zeros((rounds,yrs,n), dtype=float)
    vols = np.zeros((rounds,yrs,n), dtype=float)
    end_vols = np.zeros((rounds, n))    
    c_bals = np.zeros((rounds, n))
    c_bals_trees = np.zeros((rounds, n))
    sdwt = np.zeros((rounds, n))
    biomass_gr = np.zeros((rounds, n))
    runoff =np.zeros((rounds, int(length/dt)), dtype=float)
    
    for r, dr in enumerate(zip(spara['ditch depth west'], spara['ditch depth 20y west'], spara['ditch depth east'], spara['ditch depth 20y east'])):   #SCENARIO loop

        dwt=spara['initial h']*np.ones(spara['n'])           
        hdr_west, hdr20y_west,hdr_east, hdr20y_east = dr                                                        # drain depth [m] in the beginning and after 20 yrs
        h0ts_west = drain_depth_development(length, hdr_west, hdr20y_west)                     # compute daily values for drain bottom boundary condition
        h0ts_east = drain_depth_development(length, hdr_east, hdr20y_east)                     # compute daily values for drain bottom boundary condition

        print ('***********************************')        
        print ('Computing canopy and soil hydrology ', length, ' days', 'scenario:', scen[r])
        stp.reset_domain()   
        pt.reset_domain()
        for d in range(length):                                                 #DAY loop
            #-------Canopy hydrology--------------------------            
            reww = rew_drylimit(dwt)                                            # for each column: moisture limitation from ground water level (Feddes-function)            
            doy = forc.iloc[d, 14]
            ta =  forc.iloc[d, 4]
            vpd = forc.iloc[d, 13]
            rg = forc.iloc[d, 8]
            par = forc.iloc[d, 10]
            prec=forc.iloc[d, 7]/86400.

            hc, lai =(hdom[d], LAI[d]) # alkuperäinen
            potinf, trfall, interc, evap, ET, transpi, efloor, MBE, SWE = cpy.run_timestep(doy, dtc, ta, prec, rg, par, vpd, 
                                                            hc=hdom[d]*cmask, LAIconif=lai*cmask, Rew=reww, beta=moss.Ree) # kaikki (käytä tätä)
            potinf, efloor, MBE2 = moss.interception(potinf, efloor)
            deltas[d] = np.mean(potinf - transpi)
            ets[d] = np.mean(efloor + transpi)       

            if d%365==0: print ('  - day #', d, ' hdom ', np.round(hc,2), ' m, ',  'LAI ', np.round(lai,2), ' m2 m-2')

            #--------Soil hydrology-----------------
            dwt, ht, roff, air_ratio, afp = stp.run_timestep(d,h0ts_west[d], h0ts_east[d], deltas[d], moss)
            dwts[r,d,:] = dwt
            hts[r,d,:] = ht            
            air_ratios[r,d,:]= air_ratio
            afps[r,d,:] = afp
            runoff[r,d] = roff
            z, peat_temperature = pt.run_timestep(ta, np.mean(SWE), np.mean(efloor))
            peat_temperatures[r,d,:] = peat_temperature
              
        t5 = pd.DataFrame(peat_temperatures[r,:,1],index=pd.date_range(start_date,periods=length))   #Peat temperature in 5 cm depth, deg C

        dfwt = pd.DataFrame(dwts[r,:,:],index=pd.date_range(start_date,periods=length))
        dfair_r = pd.DataFrame(air_ratios[r,:,:], index = pd.date_range(start_date,periods=length))
        dfafp = pd.DataFrame(afps[r,:,:],index=pd.date_range(start_date,periods=length))
        v, leaf_mass, hc, b = vol[0]*np.ones(n), bmToLeafMass(bm[0])*np.ones(n), hdom[0]*np.ones(n), bm[0]*np.ones(n)
        b_ini = yiToBm(v) #b.copy()
        b = b_ini.copy()
        ageyrs = ageSim + np.array(range(yrs))+1

        start = 0            
        litter_cumul = np.zeros(n)
        Nrelease = np.zeros(n)    
        Prelease = np.zeros(n)
        Krelease = np.zeros(n)
        Crelease = np.zeros(n)
        Nleach = np.zeros(n)
        Pleach = np.zeros(n)
        Kleach = np.zeros(n)
        DOCleach = np.zeros(n)
        
        bm_deadtrees = np.zeros(n)
        stems = bmToStems(b_ini)                                                #current number of stems in the stand  (ha-1)
        n_deadtrees = np.maximum((stems - bmToStems(b)), np.zeros(n))
       
        #Ndepo = 4.0; Pdepo = 0.1; Kdepo = 0.5   #kg/ha/a Saarinen & Silfer 2011 Suo62(1):13-29, Piirainen 2002
        
        for yr in range(start_yr, end_yr+1):        
            #_, co2, Rhet, Rhet_root = heterotrophic_respiration_yr(forc, yr, dfwt, dfair_r, v, spara, dens= np.mean(spara['bd top'])) #Rhet in kg/ha/yr CO2            
            #_, co2, Rhet, Rhet_root = heterotrophic_respiration_yr(forc, yr, dfwt, dfair_r, v, spara) #Rhet in kg/ha/yr CO2            
            _, co2, Rhet, Rhet_root = heterotrophic_respiration_yr(t5, yr, dfwt, dfair_r, v, spara) #Rhet in kg/ha/yr CO2            
            days = len(co2) 
            co2release[r,start:start+days] = co2                               # mean daily time series for co2 efflux kg/ ha/day CO2
            Ns,Ps,Ks=nutrient_release(spara['sfc'],Rhet_root, N=spara['peatN'], P=spara['peatP'], K=spara['peatK']) # N P K release in kg/ha/yr                              #supply of N,P,K kg/ha/timestep
            Nstot, Pstot, Kstot = nutrient_release(spara['sfc'],Rhet, N=spara['peatN'], P=spara['peatP'], K=spara['peatK'])
            
            DOCperCO2 = 0.17   #Lauren et al 2019, Table 3 gamma CO2 for wet peat with enchytraeids 
            Nleach = Nleach + Nstot - Ns
            Pleach = Pleach + Pstot-Ps
            Kleach = Kleach + Kstot-Ks
            DOCleach = DOCleach + ((Rhet-Rhet_root)*12./44. * DOCperCO2)

            Ns, Ps, Ks = Ns+spara['depoN'], Ps+spara['depoP'], Ks+spara['depoK']        #decomposition + deposition from Ruoho-Airola et al 2003 Fig.4
            Nrelease = Nrelease + Ns
            Prelease = Prelease + Ps
            Krelease = Krelease + Ks
            Crelease = Crelease + Rhet*(12./44)                                 # CO2 to C, annual sum, nodewise in kg C ha-1
            NPP, NPP_pot = assimilation_yr(photopara, forc[str(yr)], dfwt[str(yr)], dfafp[str(yr)], leaf_mass, hc, species = spara['species'])     # NPP nodewise, kg organic matter /ha /yr sum over the year

            bm_change =  NPP - n_deadtrees/stems * b - bmToLitter(b)*365.
            bm_change_pot =  NPP_pot - n_deadtrees/stems * b - bmToLitter(b)*365.
            new_bm = b + np.maximum(bm_change, np.zeros(n))                    # suggestion for new biomass kg/ha
            new_bm_pot = b + np.maximum(bm_change_pot, np.zeros(n))
            g_npp=bmToYi(new_bm)                                               # suggested bm to new volume as yield m3/ha
            g_npp_pot = bmToYi(new_bm_pot)

            g_npps[r,yr-start_yr,:] = g_npp
            g_npps_pot[r,yr-start_yr,:] = g_npp_pot
            g_N, g_P, g_K = nut_to_vol(v, Ns,Ps,Ks,bmToLitter(b)*365., nup_gv/yrs, pup_gv/yrs, 
                                       kup_gv/yrs, leaf_mass*1000, gv_leafmass )              # volume growth allowed by nutrient release litter here in kg/ha/yr
            
            #print (np.mean(Ns), np.mean(Ps), np.mean(Ks))
            #print ('Nut limits',np.mean(g_N), np.mean(g_P), np.mean(g_K))
            #print ('***************************')
            #print ('gv leafmass, leafmass', np.mean(gv_leafmass), np.mean(leaf_mass*1000.), np.mean(gv_leafmass/(gv_leafmass + leaf_mass*1000.)))
            #print ('**************************')
            
            lim_nut_gr = np.minimum(g_K, g_P)                                  # find the growth limiting factor
            lim_nut_gr = np.minimum(lim_nut_gr, g_N)       
            #g_nut = v + lim_nut_gr
            g_nuts[r,yr-start_yr,:] = lim_nut_gr               
            v = np.minimum(lim_nut_gr, g_npp)                                  # new volume as yield 
            #print (np.mean(lim_nut_gr), np.mean(g_npp))
            
            #vols[r,yr-start_yr,:] = v 
            vols[r,yr-start_yr,:] = v 
            
            bm_restr = yiToBm(v)                                                
            
            leaf_mass, hc, b = bmToLeafMass(bm_restr), bmToHdom(bm_restr), bm_restr   
            litter_cumul = litter_cumul + bmToLitter(b)*365   
            n_deadtrees = np.maximum((stems - bmToStems(b)), np.zeros(n))
            bm_deadtrees = bm_deadtrees + n_deadtrees/stems * b
            stems = bmToStems(b)
            start = start+days    
            """
            year = yr-start_yr+1
            print (yr, '*****************')
            print ('CO2 efflux', np.mean(Rhet))
            print ('C efflux', np.mean(Rhet)*12./44)
            print ('Hdom', np.mean(hc))
            print ('v', np.mean(v) )
            print ('NPP', np.mean(NPP))
            print ('bm', np.mean(b))
            print ('b_ini', np.mean(b_ini))
            print ('bm growth', (np.mean(b) - np.mean(b_ini))/year)
            print ('litter', np.mean(bmToLitter(b)*365.))
            print ('gv_litter',  np.mean(litterfall_gv)/yrs)
            print ('dead tr', np.mean(n_deadtrees/stems * b))
            print ('delta bm', (np.mean(b) - np.mean(b_ini))/year)
            print ('C balance with trees', (np.mean((b-b_ini)/year) +  np.mean(bmToLitter(b)*365. + np.mean(n_deadtrees/stems * b) + np.mean(litterfall_gv)/yrs))*0.5  - np.mean(Rhet)*12./44. )
            print ('C balance', (np.mean(bmToLitter(b)*365. + np.mean(n_deadtrees/stems * b) + np.mean(litterfall_gv)/yrs))*0.5  - np.mean(Rhet)*12./44. )
            print ('bm gr', (np.mean((b-b_ini)/year) +  np.mean(bmToLitter(b)*365. + np.mean(n_deadtrees/stems * b))) )
            print ('Nleach', np.mean(Nleach)/year)
            print ('Pleach', np.mean(Pleach)/year)
            print ('Kleach', np.mean(Kleach)/year)
            print ('DOC leach', np.mean(Rhet-Rhet_root)*12./44. * 0.17)
            print ('*******************************************')
            """
        # Carbon balance of the stand in kg / ha / simulation time
        end_vols[r, :] = v
        c_bals_trees[r, :] = ((b-b_ini) + litter_cumul + bm_deadtrees + litterfall_gv) * 0.5 - Crelease     # in kg C /ha/time, 600 is the contribution of ground vegetation (Minkkinen et al. 20018)
        c_bals[r, :] = (litter_cumul + bm_deadtrees + litterfall_gv) * 0.5 - Crelease     # in kg C /ha/time, 600 is the contribution of ground vegetation (Minkkinen et al. 20018)
        biomass_gr[r, :] = (b-b_ini) + litter_cumul + bm_deadtrees

        v_start = vol[0]*np.ones(n)
        potential_gr = []; phys_restr=[]; chem_restr=[]
        for yr in range(start_yr, end_yr+1):   
            vgrowth = vols[0,yr-start_yr, :]-v_start
            pot_growth = g_npps_pot[0, yr-start_yr, :]-v_start
            phys_growth = g_npps[0, yr-start_yr, :]-v_start
            chem_growth = g_nuts[0, yr-start_yr, :]-v_start
            v_start=vols[0,yr-start_yr, :]
            
            potential_gr.append(np.mean(pot_growth))
            phys_restr.append(np.mean(phys_growth)/np.mean(pot_growth))
            chem_restr.append(np.mean(chem_growth)/np.mean(pot_growth))

        print ('***********restrictions*********************')
        print (np.mean(np.array(potential_gr)), np.mean(np.array(phys_restr)),np.mean(np.array(chem_restr)))
        
        #if outpara['figs']:
        #    susi_io.fig_stand_growth_node(r, rounds, ageSim, start_yr, end_yr, ageToVol, 
        #            ageyrs,g_npps[:,:, 1:-1], g_nuts[:,:, 1:-1], spara['scenario name'][r])    
        
        
        #daily_dwt = [np.mean(dwts[r,d,1:-1]) for d in range(length)] 
 
        # ******** Outputs *************************************

        susi_io.print_scenario_nodes(r,c_bals_trees, deltas, ets, h0ts_west, h0ts_east, dwts, bmToYi, g_nuts, end_vols, yi )         
        co2_respi.append(np.mean(np.sum(het[r,:,1:-1], axis=0)))
            

        summer_mean_dwt, summer = susi_io.output_dwt_growing_season(dwts[r,:], length, start_yr, end_yr, start_date, outpara, wpara, scen[r])
        sdwt[r,:] = summer_mean_dwt
        

        if outpara['to_file']: 
            susi_io.dwt_to_excel(dwts[r,:,:],outpara, scen[r])
            susi_io.runoff_to_excel(runoff[r,:]*1000., outpara, scen[r])
            #susi_io.write_excel(wlocation, wpara, spara, outpara, LAI, hdom, h0ts_west[0], h0ts_east[0],summer, summer_median_dwt)
        
        if outpara['hydfig']: susi_io.fig_hydro(stp.ele, hts[r,:,:], spara, wpara, wlocation, ets*1000., forc['Prec'].values, \
                            forc['T'].values, co2release[r,:], LAI, hdr_west, hdr_east, runoff[r,:], scen[r])

    
    #if outpara['figs']: susi_io.outfig(summer_dwt, co2_respi, growths-growths[0],spara['ditch depth'], relative_response, rounds)
    if outpara['figs']:
        susi_io.fig_stand_growth_node(rounds, ageSim, start_yr, end_yr, ageToVol, 
                    ageyrs,vols[:,:, 1:-1], spara['scenario name'], dwts)
    
    #These are for MESE-simulations only until return
    opt_strip=True   #True: normal simulation, False: mese-simulations
    if opt_strip or kaista is None:
        fr = 1 ; to = -1; frw = 1; tow  = -1        
    else:
        if kaista == 3:
            fr = int(n/2-3) ; to = int(n/2+3)
            frw = int(n/2-1) ; tow = int(n/2+1)
            
        else: 
            fr = 2 ; to = 7; frw = 2; tow  = 3        
        
        dwt_loc = np.mean(np.median(dwts[r, -245:-62,:], axis = 0)[frw:tow])
        cb = np.mean(c_bals[:,fr:to])
        cbt= np.mean(c_bals_trees[:, fr:to])
    
    #Change yield volumes to standing volumes
    
    end_vols = yiToVol(end_vols) 
    
    gr = np.array([end_vols[k]-end_vols[0] for k in range(rounds)])
    gr = np.array([np.mean(gr[k][fr:to]) for k in range(rounds)])        
    cbt = np.array([c_bals_trees[k]-0.0 for k in range(rounds)])
    cbt = np.array([np.mean(cbt[k][fr:to]) for k in range(rounds)]) 
    dcbt = np.array([c_bals_trees[k]-c_bals[0] for k in range(rounds)])
    dcbt = np.array([np.mean(dcbt[k][fr:to]) for k in range(rounds)]) 
    cb = np.array([c_bals[k]-0.0 for k in range(rounds)])
    cb = np.array([np.mean(cb[k][fr:to]) for k in range(rounds)]) 
    dcb = np.array([c_bals[k]-c_bals[0] for k in range(rounds)])
    dcb = np.array([np.mean(dcb[k][fr:to]) for k in range(rounds)]) 
    w =  np.array([sdwt[k]-0.0 for k in range(rounds)])
    w = np.array([np.mean(w[k][fr:to]) for k in range(rounds)])                
    dw =  np.array([sdwt[k]-sdwt[0] for k in range(rounds)])
    dw = np.array([np.mean(dw[k][fr:to]) for k in range(rounds)])        
    v_end = np.array([end_vols[k]-0. for k in range(rounds)])
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

    mese_single = False
    mese_multiple = True
    vesitase=False
    generic_run = False
    gui = False
    if gui:
        return v[0], v_end, (v_end-vol[0])/yrs, cbt, dcbt, cb, dcb,  w, dw, logs, pulp, dv, dlogs, dpulp, yrs, bms/yrs
        #        dw, logs, pulp, dv, dlogs, dpulp, yrs, bms/yrs
    if vesitase:  #-1  #0 for mese, -1 for vesitase
        #This output when single scenario is computed
        #return (vol[0], v_end, gr/yrs, cbt/yrs, dcbt/yrs, cb/yrs, dcb/yrs, w, \
        #        dw, logs, pulp, dv, dlogs, dpulp, yrs, bms/yrs, \
        #        np.mean(np.array(potential_gr)), np.mean(np.array(phys_restr)),np.mean(np.array(chem_restr)))
        print ('1111111111111111111111')
       
        return (vol[0], v_end, np.mean((v_end-vol[0]))/yrs, cbt/yrs, dcbt/yrs, cb/yrs, dcb/yrs, w, \
                dw, logs, pulp, dv, dlogs, dpulp, yrs, bms/yrs, \
                np.mean(np.array(potential_gr)), np.mean(np.array(phys_restr)),np.mean(np.array(chem_restr)),
                np.mean(Nleach)/yrs, np.mean(Pleach)/yrs, np.mean(Kleach)/yrs, np.mean(DOCleach)/yrs, annual_runoff)
                
    if mese_single:    
        return (vol[0], np.mean(v[fr:to]), np.mean((v-vol[0])[fr:to])/yrs,  np.mean(Nrelease[fr:to]), 
                np.mean(Prelease[fr:to]), np.mean(Krelease[fr:to]), np.mean(Crelease[fr:to])/yrs, dwt_loc, 
                cb/yrs, cbt/yrs,annual_runoff, drunoff, w, dw, dv)

    if mese_multiple:
        return (vol[0], v_end, gr, w, dw, dv)
        
    if generic_run:   
        return (vol[0], v_end, (v_end-vol[0])/yrs,  Nrelease[fr:to], 
                Prelease[fr:to], Krelease[fr:to], Crelease[fr:to]/yrs, dwt_loc, cb/yrs, 
                cbt/yrs, annual_runoff, drunoff, w, dw, dv)
       

          