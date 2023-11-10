import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress
from matplotlib.ticker import MultipleLocator
from ase.db import connect
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

# Get adsorption energies on single element
db_path = "../databases"
with connect(f'{db_path}/single_element_slabs_out.db') as db_slab,\
    connect(f'{db_path}/single_element_H_out.db') as db_H,\
    connect(f'{db_path}/single_element_CO_out.db') as db_CO,\
    connect(f'{db_path}/single_element_NO_fcc_out.db') as db_NO,\
    connect(f'{db_path}/molecules_out.db') as db_gas:
        
    # Get No and H gas phase energies
    E_NO = db_gas.get(molecule='NO').energy + 0.29
    E_H = 0.5*db_gas.get(molecule='H2').energy

    # Get CO on Cu reference energy
    E_ref_CO = db_CO.get(metal='Cu').energy - db_slab.get(metal='Cu').energy + 0.12
    
    # Get single metal energies
    H_metal_energies=[db_H.get(metal=m).energy - db_slab.get(metal=m).energy - E_H  for m in metals]
    CO_metal_energies=[db_CO.get(metal=m).energy - db_slab.get(metal=m).energy - E_ref_CO - G_corr['CO'] for m in metals]
    NO_metal_energies=[db_NO.get(metal=m).energy - db_slab.get(metal=m).energy - E_NO for m in metals]



# Load data, skip pure metals
CO_data = np.loadtxt("../features/CO.csv",delimiter=',',skiprows=6,usecols=(0,1,2,3,4,20,22,23))
NO_df = pd.read_csv("../features/NO_fcc.csv",sep=',',header=None, skiprows=6)
H_df = pd.read_csv("../features/H.csv",sep=',',header=None, skiprows=6)

# Get top site metals for CO
CO_site_metal = np.array([metals[i] for i in np.nonzero(CO_data[:,:5])[1]])
# Get CO energies, slab ids and the adsorbed site id on the slab
CO_energies, CO_slabs_ids, CO_site_ids = CO_data[:,-3], CO_data[:,-2], CO_data[:,-1]

# Get ensembles
n_atoms_ensemble = 3
ensembles = list(it.combinations_with_replacement(metals, n_atoms_ensemble))

# Get NO site ensembles, energies, slab ids and ids of surface atoms in ensemble site
NO_site_ensembles = [ensembles[i] for i in np.nonzero(NO_df.iloc[:,:35].to_numpy())[1]]
NO_energies, NO_slabs_ids = NO_df[45].to_numpy(), NO_df[47].to_numpy()
NO_site_ids = np.array([ast.literal_eval(site_ids.replace(' ',',')) for site_ids in NO_df[48]])

# Get H site ensembles, energies, slab ids and ids of surface atoms in ensemble site
H_site_ensembles = [ensembles[i] for i in np.nonzero(H_df.iloc[:,:35].to_numpy())[1]]
H_energies, H_slabs_ids = H_df[45].to_numpy(), H_df[47].to_numpy()
H_site_ids = np.array([ast.literal_eval(site_ids.replace(' ',',')) for site_ids in H_df[48]])


# Correct to adsorption energies
NO_energies-=G_corr['NO']
CO_energies-=G_corr['CO']
H_energies-=G_corr['H']

# Make plot
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(13,4))
count=0
for metal in metals:
    # Mask of where CO site is the current metal
    CO_mask = CO_site_metal==metal
    
    # Go through each CO energy, slab id and site id with applied mask
    for CO_energy, slab_id, site_id in zip(CO_energies[CO_mask],CO_slabs_ids[CO_mask],CO_site_ids[CO_mask]):
        # Get slab and site mask 
        H_slab_mask = H_slabs_ids==slab_id
        H_site_mask = np.any(H_site_ids==site_id,axis=1)
        # Get mask of where both slab and CO site is in H ensemble
        H_mask = H_slab_mask * H_site_mask
        # Apply mask to obtain H energy
        H_energies_masked = H_energies[H_mask]
        # Plot all H energies that have the CO site in the ensemble
        ax1.scatter(H_energies_masked,np.full(len(H_energies_masked),CO_energy),c=metal_colors[metal],marker='.')

        # Get slab and site mask 
        NO_slab_mask = NO_slabs_ids==slab_id
        NO_site_mask = np.any(NO_site_ids==site_id,axis=1)
        # Get mask of where both slab and CO site is in NO ensemble
        NO_mask = NO_slab_mask * NO_site_mask
        # Apply mask to obtain NO energy
        NO_energies_masked = NO_energies[NO_mask]

        # Check if CO site would be blocked by H
        if np.any(H_energies_masked<=(-G_corr['H'])):
            # Plot datapoint as unfilled marker if CO is blocked by H
            ax3.scatter(np.full(len(NO_energies_masked),CO_energy), NO_energies_masked,edgecolors=metal_colors[metal],marker='.',facecolors='none',linewidth=1)
            count+=1
        else:
            ax3.scatter(np.full(len(NO_energies_masked),CO_energy), NO_energies_masked,c=metal_colors[metal],marker='.')

    
# Array to store NO and H energies for regression
H_NO_energies = np.empty((0,2))
# Iterate through each NO data property
for NO_energy, slab_id, site_id, NO_ensemble in zip(NO_energies, NO_slabs_ids, NO_site_ids,NO_site_ensembles):

    # mask of where H is adsorped on the same site on the same slab
    mask = (slab_id == H_slabs_ids) * (np.all(site_id==H_site_ids,axis=1))

    # Plot datapoint if both are adsorbed to the site
    if np.any(mask):
        # Get energy and plot
        H_energy = np.mean(H_energies[mask])
        ax2.scatter(H_energy,NO_energy,color=metal_colors[''.join(NO_ensemble)],marker='.')
        # Store energies
        H_NO_energies = np.vstack((H_NO_energies,[H_energy,NO_energy]))


# plot single metals
single_metal_colors = [metal_colors[metal] for metal in metals]
ax1.scatter(H_metal_energies,CO_metal_energies,color=single_metal_colors,edgecolors='k',alpha=0.7)
ax2.scatter(H_metal_energies,NO_metal_energies,color=single_metal_colors,edgecolors='k',alpha=0.7)
ax3.scatter(CO_metal_energies,NO_metal_energies,color=single_metal_colors,edgecolors='k',alpha=0.7)

# Linear regression
a,b, *ignore= linregress(H_NO_energies.T[0],H_NO_energies.T[1])

# Get axes limits
xlim1, ylim1 = ax1.get_xlim(), ax1.get_ylim()
xlim2, ylim2 = ax2.get_xlim(), ax2.get_ylim()
xlim3, ylim3 = ax3.get_xlim(), ax3.get_ylim()

# Plot dG=0 lines
ax1.vlines(-G_corr['H'], ylim1[0],ylim1[1],color='k',ls='--',alpha=0.6,lw=1.5)
ax1.hlines(-G_corr['CO'], xlim1[0],xlim1[1],color='k',ls='--',alpha=0.6,lw=1.5)
ax1.set_xlim(*xlim1)
ax1.set_ylim(*ylim1)

# Plot scaling relation
xarr = np.array([xlim2[0],xlim2[1]])
ax2.plot(xarr,a*xarr + b,c='b',ls='--',label='Fit')

# Plot dG=0 lines
ax2.vlines(-G_corr['H'], ylim2[0],ylim2[1],color='k',ls='--',alpha=0.6,lw=1.5)
ax2.hlines(-G_corr['NO'], xlim2[0],xlim2[1],color='k',ls='--',alpha=0.6,lw=1.5)
ax2.set_xlim(*xlim2)
ax2.set_ylim(*ylim2)
# Scaling text
ax2.text(-G_corr['H']+.02,-1.7,'\u0394$E_{DFT}^{*NO} = $' + f'{a:.2f}' + '\u0394$E_{DFT}^{*H}$' + f'\u2212{abs(b):.2f} eV',fontsize=9,color='b',va='center',ha='left')

ax2.legend(loc=2)

# Plot dG=0 lines
ax3.vlines(-G_corr['CO'], ylim3[0],ylim3[1],color='k',ls='--',alpha=0.6,lw=1.5)
ax3.hlines(-G_corr['NO'], xlim3[0],xlim3[1],color='k',ls='--',alpha=0.6,lw=1.5)
ax3.hlines(a*(-G_corr['H'])+b, xlim3[0],xlim3[1],color='b',ls='--',lw=1.5)
ax3.set_xlim(*xlim3)
ax3.set_ylim(*ylim3)

ax1.set_xlabel("\u0394$E_{DFT}^{*H}$ [eV]")
ax1.set_ylabel("\u0394$E_{DFT}^{*CO}$ [eV]")

ax2.set_xlabel("\u0394$E_{DFT}^{*H}$ [eV]")
ax2.set_ylabel("\u0394$E_{DFT}^{*NO}$ [eV]")

ax3.set_xlabel("\u0394$E_{DFT}^{*CO}$ [eV]")
ax3.set_ylabel("\u0394$E_{DFT}^{*NO}$ [eV]")

for i,ax in enumerate([ax1,ax2,ax3]):
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

# Make label handles
handles = []
for metal in metals:
    handles.append(Line2D([0], [0], marker='o', color="w", label=metal,markerfacecolor=metal_colors[metal], markersize=12)) 

plt.tight_layout()

# Axes positions for legend placement
pos1 = ax1.get_position()
pos3 = ax3.get_position()
# Create legend on top of subplots
fig.legend(handles=handles, labels=metals,
           loc='outside upper center', ncol=5, mode='expand',fontsize=12,bbox_to_anchor=(pos1.x0, .5, pos3.x1-pos1.x0, 0.5),fancybox=False)

# Make room for legend on top
fig.subplots_adjust(top=0.89)
plt.savefig('DFT_scaling.png', dpi=600)