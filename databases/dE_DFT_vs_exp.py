import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect

import sys
sys.path.append('..')
from scripts import metal_colors


metal_colors['Rh'] = 'darkcyan'
metal_colors['Ir'] = 'midnightblue'
metal_colors['Ni'] = 'limegreen'


E_slab = {}
with connect('single_element_slabs_out.db') as db1, connect('single_element_slabs_extra_out.db') as db2:
    for db in [db1,db2]:
        for row in db.select():
            
            E_slab[row.metal] = row.energy


E_tot_CO = {}
with connect('single_element_CO_out.db') as db:
    for row in db.select():
            
            if row.metal == 'Pd':
                 pass
            else:
                E_tot_CO[row.metal] = row.energy

E_tot_NO = {}
with connect('single_element_NO_fcc_out.db') as db:
     for row in db.select():
            
            E_tot_NO[row.metal] = row.energy



with connect('single_element_ads_extra_out.db') as db:
     for row in db.select():
        if row.id<5:
               
            E_tot_CO[row.metal] = row.energy
        else:
            E_tot_NO[row.metal] = row.energy

E_ads = {}
with connect('molecules_out.db') as db:
     for row in db.select():
            
            E_ads[row.molecule] = row.energy
E_ads['NO'] += 0.29

CO_exp = {
     'Ag': [-0.28],
     'Au': [-0.40],
     'Cu': [-0.51,-0.57],
     'Pd': [-1.34,-1.38,-1.41,-1.43,-1.43,-1.46,-1.47,-1.57],
     'Pt': [-1.20,-1.22,-1.26,-1.28,-1.37],
     'Rh': [-1.29,-1.33,-1.60],
     'Ir': [-1.58],
     'Ni': [-1.28],
}

NO_exp = {
     'Pd': [-1.81],
     'Pt': [-1.16],
     'Rh': [-1.17,-1.48]
}



fig,ax = plt.subplots()

errors = []
dEs = []
diffs = []
exps = []


metals = ['Ag','Au','Cu','Pd','Pt','Rh','Ir','Ni']

for metal in metals:
     
    exp = CO_exp[metal]

    dE = E_tot_CO[metal] - E_slab[metal] - E_ads['CO']

    if len(exp)>1:
        exp_mean = np.mean(exp)
        exp_err = np.std(exp,ddof=1)
        
        
    elif metal == 'Ir' or metal=='Ni':
        exp_err = 0.13
        exp_mean = exp[0]

    else:
        exp_err = 0.2
        exp_mean = exp[0]

    diff = dE-exp_mean
    ax.errorbar(dE,diff,yerr=exp_err,label=metal,fmt='.',color=metal_colors[metal])
    errors.append(exp_err)
    dEs.append(dE)
    diffs.append(diff)
    exps.append(exp_mean)


for metal in ['Pd','Pt','Rh']:
    exp = NO_exp[metal]

    dE = E_tot_NO[metal] - E_slab[metal] - E_ads['NO']

    if len(exp)>1:
        exp_mean = np.mean(exp)
        exp_err = np.std(exp,ddof=1)
    else:
        exp_err = 0.2
        exp_mean = exp[0]

    diff = dE-exp_mean
    ax.errorbar(dE,diff,yerr=exp_err,marker='x',color=metal_colors[metal])

    errors.append(exp_err)
    dEs.append(dE)
    diffs.append(diff)
    exps.append(exp_mean)



from scipy.optimize import curve_fit

def line(x,a,b):
     return a*x + b

dEs = np.array(dEs)

(a,b), pcov = curve_fit(line,dEs,diffs,sigma=errors)

residuals = diffs- line(dEs, a,b)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((diffs-np.mean(diffs))**2)
R2 = 1 - (ss_res / ss_tot)


xlim = np.array(ax.get_xlim())
ax.plot(xlim,a*xlim + b,c='k')

ax.text(0.25,0.9,f'$R^2 = {R2:.3f}$\nFit: Error $= {a:.2f}\Delta E_{{DFT}} + {b:.2f}$',transform=ax.transAxes)

ax.legend()

ax.set_ylabel('Error: $\Delta E_{DFT} - \Delta E_{exp}$ [eV]')
ax.set_xlabel('$\Delta E_{DFT}$ [eV]')

# h,l = ax.get_legend_handles_labels()

plt.tight_layout()
# plt.savefig('dE_CO_DFT_vs_ecp.png',dpi=600)
plt.savefig('dE_CO_NO_DFT_vs_ecp.png',dpi=600)

# plt.show()
plt.close()

errors[0:2]=[0,0]
errors[-3:-1] = [0,0]
dEs_corr = dEs - (a*dEs + b)


fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(8,4))


for i, metal in enumerate(metals):
    ax1.errorbar(exps[i],dEs[i],xerr=errors[i],color=metal_colors[metal],fmt='.',label=metal)
    ax2.errorbar(exps[i],dEs_corr[i],xerr=errors[i],color=metal_colors[metal],fmt='.',label=metal)
    # print(metal,dEs_corr[i],dEs_corr[i]+0.4)

for i, metal in enumerate(['Pd','Pt','Rh']):
    i+=len(metals)
    ax1.errorbar(exps[i],dEs[i],xerr=errors[i],color=metal_colors[metal],fmt='.',label=metal,marker='x')
    ax2.errorbar(exps[i],dEs_corr[i],xerr=errors[i],color=metal_colors[metal],fmt='.',label=metal,marker='x')
    # print(metal,dEs_corr[i],dEs_corr[i]+0.556)

# lim1 = ax1.get_ylim()

lim=[-2.3,0.1]

ax1.plot(lim,lim,c='k')

ax1.set_xlim(*lim)
ax1.set_ylim(*lim)



# lim2 = ax2.get_ylim()
ax2.plot(lim,lim,c='k')

ax2.set_xlim(*lim)
ax2.set_ylim(*lim)


ax1.set_xlabel('$\Delta E_{exp}$ [eV]')
ax2.set_xlabel('$\Delta E_{exp}$ [eV]')

ax1.set_ylabel('$\Delta E_{DFT}$ [eV]')
ax2.set_ylabel('$\Delta E_{DFT,corrected}$ [eV]')

ax1.legend()
ax2.legend()

plt.tight_layout()
# plt.savefig('dE_CO_corr.png',dpi=600)
plt.savefig('dE_CO_NO_corr.png',dpi=600)
# plt.show()



print('CO:')     
for metal in ['Ag','Au','Cu','Pd','Pt']:
     dE = E_tot_CO[metal] - E_slab[metal] - E_ads['CO']
     dE_corr = dE - (a*dE + b)
     print(metal, dE_corr,dE_corr + 0.4)

print('NO:')
for metal in ['Ag','Au','Cu','Pd','Pt']:
     dE = E_tot_NO[metal] - E_slab[metal] - E_ads['NO']
     dE_corr = dE - (a*dE + b)
     print(metal, dE_corr,dE_corr + 0.556)
