# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:34:55 2020

@author: alauren
"""
#%%
from scipy.sparse import  diags
from scipy.sparse.linalg import  spsolve
import numpy as np
import matplotlib.pylab as plt
#%%

def my_sin( x, freq, amplitude, phase, offset):
    # create the function we want to fit
    return np.sin(x * freq + phase) * amplitude + offset


#%%
def stand_alone():
    days = 1                    # lenght of simulation time, days
    L=3.5                       # depth of profile, m
    Nx =int(20*L)               # number of soil layers in the profile (0.05 m each)
    T =86400*days               # simulation time in seconds, s
    Nt = 12*days                # number of time points in simulation
    ini = np.ones(Nx+1)*3.      # initial temperature in the profile, ddeg C
    
    x = np.linspace(0, L, Nx+1)    # mesh points in space
    dx = x[1] - x[0]
    
    t = np.linspace(0, T, Nt+1)    # mesh points in time
    dt = t[1] - t[0]
    
    D = 1e-7                    #Thermal diffusivity of peat, m2 s-1, de Vries 1975
    F = D*dt/dx**2   
    u   = np.zeros(Nx+1)           # unknown u at new time level
    u_1 = np.zeros(Nx+1)           # u at the previous time level
    
    # Representation of sparse matrix and right-hand side
    main  = np.zeros(Nx+1)
    lower = np.zeros(Nx)
    upper = np.zeros(Nx)
    b     = np.zeros(Nx+1)
    
    # Precompute sparse matrix
    main[:] = 1 + 2*F
    lower[:] = -F  #1
    upper[:] = -F  #1
    # Insert boundary conditions, Diritchlet (constant value) boundaries
    main[0] = 1
    main[Nx] = 1
    
    # Create the main matrix
    A = diags(
        diagonals=[main, lower, upper],
        offsets=[0, -1, 1], shape=(Nx+1, Nx+1),
        format='csr')
    #print (A.todense())  # Check that A is correct
    
    # Set initial condition
    #for i in range(0,Nx+1):
    #    u_1[i] = I(x[i])
    u_1 = ini.copy()
    
    freq = 10 
    amplitude = 3.   #aallonkorkeus
    phase = 10. 
    offset = 5.  #mean
    temp = my_sin( t, freq, amplitude, phase, offset)
    fig = plt.figure(num='temperature')
    plt.subplot(211)
    plt.plot(t, temp)
    
    plt.subplot(212)
    plt.xlim([0,20.])
    for n in range(0, Nt):
        b = u_1
        b[0] = temp[n] #0.0  # boundary conditions
        b[-1] = 3.
        u[:] = spsolve(A, b)
        u_1[:] = u
    
        plt.plot(u[:30],-x[:30])

#stand_alone()
#%%
def run_soil_temperature(Tsoil, Tair):
    """
    Computes daily temperature profile in peat soil
    Input: 
        Tsoil, temperature profile in previous time step, deg C
        Tair, air temperature in deg C
    Returns
    -------
    soil temperature in layers in deg C
    """
    #Soil related parameters: 
    L=3.5                                           # depth of profile, m
    Nz =int(20*L)                                   # number of soil layers in the profile (0.05 m each)
    z =   np.linspace(0, L, Nz+1)                   # mesh points in space       
    dz = z[1] - z[0]
    
    T = 86400                                       # timestep in seconds, s
    Nt = 12                                         # number of subtimesteps in the time step (here every 2 hrs)
    t = np.linspace(0, T, Nt+1)                     # mesh points in time
    dt = t[1] - t[0]
    
    D = 1e-7                                        # Thermal diffusivity of peat, m2 s-1, de Vries 1975
    F = D*dt/dz**2   
    u   = np.zeros(Nz+1)                            # unknown u at new time level
    u_1 = Tsoil                                     # soil temperature at previous timestep degC
    
    
    # Representation of sparse matrix and right-hand side
    main  = np.zeros(Nz+1)
    lower = np.zeros(Nz)
    upper = np.zeros(Nz)
    b     = np.zeros(Nz+1)
    
    # Precompute sparse matrix
    main[:] = 1 + 2*F
    lower[:] = -F  #1
    upper[:] = -F  #1
    # Insert boundary conditions, Diritchlet (constant value) boundaries
    main[0] = 1
    main[Nz] = 1
    
    # Create the main matrix
    A = diags(
        diagonals=[main, lower, upper],
        offsets=[0, -1, 1], shape=(Nz+1, Nz+1),
        format='csr')
    #print (A.todense())  # Check that A is correct

    for n in range(0, Nt):
        b = u_1
        b[0] = Tair #temp[n] #0.0  # boundary conditions
        b[-1] = 3.
        u[:] = spsolve(A, b)
        u_1[:] = u
    return z, u


fig = plt.figure(num='temperature')
plt.subplot(111)
Tsoil = np.ones(int(20*3.5)+1)*5.

for n in range(10):
    z, Tsoil = run_soil_temperature(Tsoil, 10.)
    plt.plot(Tsoil,-z)
