# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:24:19 2020

@author: alauren
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#%%
folder = r'C:/Users/alauren/Documents/WinPython-64bit-2.7.10.3/Susi_8_3_py37/outputs/sensitivity/'

def get_vals(f):
    df = pd.read_excel(folder+f)
    wtmean = df['wt'].mean()
    wtsd = df['wt'].std()
    ivmean = df['iv5'].mean()
    ivsd = df['iv5'].std()
    potgrmean = df['pot_gr'].mean()
    potgrsd = df['pot_gr'].std()
    physmean = df['phys_r'].mean()
    physsd = df['phys_r'].std()
    chemmean = df['chem_r'].mean()
    chemsd = df['chem_r'].std()
    return wtmean, wtsd, ivmean, ivsd, potgrmean, potgrsd, physmean, physsd, chemmean, chemsd     


#%%


wtbas, wtsdbas, ivbas, ivsdbas, potgrbas, potgrsdbas, physbas, physsdbas, chembas, chemsdbas = get_vals('vols_basic.xlsx')

fig = plt.figure(num='sensitivity', figsize=(10,6))


y = 0.96
ax0 = fig.add_axes([0.2, 0.15, 0.1, 0.75]) #left, bottom, width, height)
plt.title('WT')
ax0.set_xlim([-0.8, -0.2])
ax0.set_ylim([0.0, 1.0])

plt.vlines(wtbas,0,1, linewidth=0.5)
ax0.axvspan(wtbas-1*wtsdbas, wtbas+1*wtsdbas, alpha=0.3, color='blue')
for ylabel_i in ax0.get_yticklabels():
    ylabel_i.set_fontsize(0.0)
    ylabel_i.set_visible(False)

ax1 = fig.add_axes([0.35, 0.15, 0.1, 0.75]) #left, bottom, width, height)
ax1.set_xlim([4.0, 9.0])
ax1.set_ylim([0.0, 1.0])

plt.vlines(ivbas,0,1, linewidth=0.5)
ax1.axvspan(ivbas-1*ivsdbas, ivbas+1*ivsdbas, alpha=0.3, color='blue')

plt.title('iv')
for ylabel_i in ax1.get_yticklabels():
    ylabel_i.set_fontsize(0.0)
    ylabel_i.set_visible(False)

ax2 = fig.add_axes([0.5, 0.15, 0.1, 0.75]) #left, bottom, width, height)
ax2.set_xlim([8.0, 18.0])
ax2.set_ylim([0.0, 1.0])

plt.vlines(potgrbas,0,1, linewidth=0.5)
ax2.axvspan(potgrbas-1*potgrsdbas, potgrbas+1*potgrsdbas, alpha=0.3, color='blue')
plt.title('pot growth')

for ylabel_i in ax2.get_yticklabels():
    ylabel_i.set_fontsize(0.0)
    ylabel_i.set_visible(False)

ax3 = fig.add_axes([0.65, 0.15, 0.1, 0.75]) #left, bottom, width, height)
ax3.set_xlim([0.5, 1.0])
ax3.set_ylim([0.0, 1.0])

plt.vlines(physbas,0,1, linewidth=0.5)
ax3.axvspan(physbas-1*physsdbas, physbas+1*physsdbas, alpha=0.3, color='blue')

plt.title('phys rest')
for ylabel_i in ax3.get_yticklabels():
    ylabel_i.set_fontsize(0.0)
    ylabel_i.set_visible(False)

ax4 = fig.add_axes([0.8, 0.15, 0.1, 0.75]) #left, bottom, width, height)
ax4.set_xlim([0.3, 0.8])
ax4.set_ylim([0.0, 1.0])

plt.vlines(chembas,0,1, linewidth=0.5)
ax4.axvspan(chembas-1*chemsdbas, chembas+1*chemsdbas, alpha=0.3, color='blue')

plt.title('chem rest')
for ylabel_i in ax4.get_yticklabels():
    ylabel_i.set_fontsize(0.0)
    ylabel_i.set_visible(False)


#rows

var =['$S_{width} $ +20 %', '$S_{width} $ -20 %','$D_{depth} $ +20%', '$D_{depth} $ -20%',  'Root layer 20 cm',
      'Root layer 40 cm', 'Afp 20 cm', 'Afp 0 cm', 'Nutr conc +20%', 'Nutr conc -20%', 'Anisotropy + 20%', 'Anisotropy + 20%', 
      'Sphagnum', 'Carex']
#ys =[0.96, 0.91, 0.86, 0.81, 0.76,0.71, 0.66, 0.61, 0.56, 0.51, 0.46, 0.41, 0.36,0.31]
ys = np.linspace(0.05,0.96, len(var))
ys=-np.sort(-ys)
files =['vols_swidth+20.xlsx', 'vols_swidth-20.xlsx', 'vols_ddepth+20.xlsx', 'vols_ddepth-20.xlsx', 
        'vols_nroot_30.xlsx', 'vols_nroot_50.xlsx',
        'vols_afp_20.xlsx', 'vols_afp_5.xlsx', 'vols_nut+20.xlsx', 'vols_nut-20.xlsx', 'vols_anisotropy+20.xlsx',
        'vols_anisotropy-20.xlsx', 'vols_sphagnum.xlsx', 'vols_carex.xlsx']

for v, y, f in zip(var, ys,files):
    wt, wtsd, iv, ivsd, potgr, potgrsd, phys, physsd, chem, chemsd = get_vals(f)

    ax0.text(-1.8, y, v, horizontalalignment='left',
           verticalalignment='top', fontsize=14, transform = ax0.transAxes)
    ax0.plot(wt, y, 'ro', markersize = 5)
    ax0.errorbar(wt, y, xerr=1*wtsd, color='red', capsize=4)

    ax1.plot(iv, y, 'ro', markersize = 5)
    ax1.errorbar(iv, y, xerr=1*ivsd, color='red', capsize=4)

    ax2.plot(potgr, y, 'ro', markersize = 5)
    ax2.errorbar(potgr, y, xerr=1*potgrsd, color='red', capsize=4)

    ax3.plot(phys, y, 'ro', markersize = 5)
    ax3.errorbar(phys, y, xerr=1*physsd, color='red', capsize=4)

    ax4.plot(chem, y, 'ro', markersize = 5)
    ax4.errorbar(chem, y, xerr=1*chemsd, color='red', capsize=4)
