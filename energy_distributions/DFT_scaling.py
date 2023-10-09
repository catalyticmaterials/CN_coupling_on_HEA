import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress
from matplotlib.ticker import MultipleLocator
import itertools as it
import pandas as pd
import ast
import sys
sys.path.append("..")
from scripts import metals, metal_colors, G_corr

plt.rc('font', size=14)
plt.rc('ytick', labelsize=12)
plt.rc('xtick', labelsize=12) 
plt.rc('legend', fontsize=12)
plt.rc('lines', linewidth=1.2,markersize=8)

# Load data, skip pure metals
CO_data = np.loadtxt("../features/CO.csv",delimiter=',',skiprows=6,usecols=(0,1,2,3,4,20,22,23))
NO_df = pd.read_csv("../features/NO_fcc.csv",sep=',',header=None, skiprows=6)
H_df = pd.read_csv("../features/H.csv",sep=',',header=None, skiprows=6)

CO_site_metal = np.array([metals[i] for i in np.nonzero(CO_data[:,:5])[1]])
CO_energies, CO_slabs_ids, CO_site_ids = CO_data[:,-3], CO_data[:,-2], CO_data[:,-1]

# Get ensembles
n_atoms_ensemble = 3
ensembles = list(it.combinations_with_replacement(metals, n_atoms_ensemble))

NO_site_ensembles = [ensembles[i] for i in np.nonzero(NO_df.iloc[:,:35].to_numpy())[1]]
NO_energies, NO_slabs_ids = NO_df[45].to_numpy(), NO_df[47].to_numpy()
NO_site_ids = np.array([ast.literal_eval(site_ids.replace(' ',',')) for site_ids in NO_df[48]])

H_site_ensembles = [ensembles[i] for i in np.nonzero(H_df.iloc[:,:35].to_numpy())[1]]
H_energies, H_slabs_ids = H_df[45].to_numpy(), H_df[47].to_numpy()
H_site_ids = np.array([ast.literal_eval(site_ids.replace(' ',',')) for site_ids in H_df[48]])


# Correct to adsorption energies
NO_energies-=G_corr['NO']
CO_energies-=G_corr['CO']
H_energies-=G_corr['H']


fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(13,4))
count=0
for metal in metals:
    CO_mask = CO_site_metal==metal
    
    for CO_energy, slab_id, site_id in zip(CO_energies[CO_mask],CO_slabs_ids[CO_mask],CO_site_ids[CO_mask]):
        
        H_slab_mask = H_slabs_ids==slab_id
        H_site_mask = np.any(H_site_ids==site_id,axis=1)
        H_mask = H_slab_mask * H_site_mask

        H_energies_masked = H_energies[H_mask]

        ax1.scatter(H_energies_masked,np.full(len(H_energies_masked),CO_energy),c=metal_colors[metal],marker='.')


        NO_slab_mask = NO_slabs_ids==slab_id
        NO_site_mask = np.any(NO_site_ids==site_id,axis=1)
        NO_mask = NO_slab_mask * NO_site_mask

        NO_energies_masked = NO_energies[NO_mask]


        if np.any(H_energies_masked<=(-0.1)):
            
            ax3.scatter(np.full(len(NO_energies_masked),CO_energy), NO_energies_masked,edgecolors=metal_colors[metal],marker='.',facecolors='none',linewidth=1)
            count+=1

        else:
            
            

            ax3.scatter(np.full(len(NO_energies_masked),CO_energy), NO_energies_masked,c=metal_colors[metal],marker='.')

    


H_NO_energies = np.empty((0,2))

for NO_energy, slab_id, site_id, NO_ensemble in zip(NO_energies, NO_slabs_ids, NO_site_ids,NO_site_ensembles):

    mask = (slab_id == H_slabs_ids) * (np.all(site_id==H_site_ids,axis=1))

    if np.any(mask):
         
        H_energy = np.mean(H_energies[mask])
    
        ax2.scatter(H_energy,NO_energy,color=metal_colors[''.join(NO_ensemble)],marker='.')

        H_NO_energies = np.vstack((H_NO_energies,[H_energy,NO_energy]))




a,b, *ignore= linregress(H_NO_energies.T[0],H_NO_energies.T[1])

xlim1, ylim1 = ax1.get_xlim(), ax1.get_ylim()
xlim2, ylim2 = ax2.get_xlim(), ax2.get_ylim()
xlim3, ylim3 = ax3.get_xlim(), ax3.get_ylim()


ax1.vlines(-G_corr['H'], ylim1[0],ylim1[1],color='k',ls='--',alpha=0.6)
ax1.hlines(-G_corr['CO'], xlim1[0],xlim1[1],color='k',ls='--',alpha=0.6)
ax1.set_xlim(*xlim1)
ax1.set_ylim(*ylim1)

# scaling_str = '\u0394$E_{DFT}^{*NO} = $' + f'{a:.2f}' + '\u0394$E_{DFT}^{*H}$' + f'\u2212{abs(b):.2f} eV'
xarr = np.array([xlim2[0],xlim2[1]])
ax2.plot(xarr,a*xarr + b,c='b',ls='--',label='Fit')

ax2.vlines(-G_corr['H'], ylim2[0],ylim2[1],color='k',ls='--',alpha=0.6)
ax2.hlines(-G_corr['NO'], xlim2[0],xlim2[1],color='k',ls='--',alpha=0.6)
ax2.set_xlim(*xlim2)
ax2.set_ylim(*ylim2)

ax2.text(-G_corr['H']+.02,-1.7,'\u0394$E_{DFT}^{*NO} = $' + f'{a:.2f}' + '\u0394$E_{DFT}^{*H}$' + f'\u2212{abs(b):.2f} eV',fontsize=9,color='b',va='center',ha='left')

ax2.legend(loc=2)

ax3.vlines(-G_corr['CO'], ylim3[0],ylim3[1],color='k',ls='--',alpha=0.6)
ax3.hlines(-G_corr['NO'], xlim3[0],xlim3[1],color='k',ls='--',alpha=0.6)
ax3.hlines(a*(-G_corr['H'])+b, xlim3[0],xlim3[1],color='b',ls='--')
ax3.set_xlim(*xlim3)
ax3.set_ylim(*ylim3)

ax1.set_xlabel("\u0394$E_{DFT}^{*H}$ [eV]")
ax1.set_ylabel("\u0394$E_{DFT}^{*CO}$ [eV]")

ax2.set_xlabel("\u0394$E_{DFT}^{*H}$ [eV]")
ax2.set_ylabel("\u0394$E_{DFT}^{*NO}$ [eV]")

ax3.set_xlabel("\u0394$E_{DFT}^{*CO}$ [eV]")
ax3.set_ylabel("\u0394$E_{DFT}^{*NO}$ [eV]")

for i,ax in enumerate([ax1,ax2,ax3]):
    # if i==2:
    #     ax.xaxis.set_major_locator(MultipleLocator(0.5))
    #     ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    # else:
    #     ax.xaxis.set_major_locator(MultipleLocator(0.2))
    #     ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
  



handles = []
for metal in metals:
    handles.append(Line2D([0], [0], marker='o', color="w", label=metal,markerfacecolor=metal_colors[metal], markersize=12)) 

plt.tight_layout()


pos1 = ax1.get_position()
pos3 = ax3.get_position()

fig.legend(handles=handles, labels=metals,
           loc='outside upper center', ncol=5, mode='expand',fontsize=12,bbox_to_anchor=(pos1.x0, .5, pos3.x1-pos1.x0, 0.5),fancybox=False)


# plt.tight_layout()
fig.subplots_adjust(top=0.89)
plt.savefig('DFT_scaling.png', dpi=600)
# plt.show()