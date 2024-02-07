import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from ase.db import connect
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from density_plot_functions import probability_density_plot
import sys
sys.path.append('..')
from scripts.methods import get_sites
from scripts import metals, G_corr, metal_colors


plt.rc('font', size=8)
plt.rc('legend', fontsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('xtick', labelsize=8) 

# Atom RGB colors from jmol
C_color = np.array([80,80,80])/255
N_color = np.array([48,80,248])/255


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


# Set compositon to be equimolar
composition = np.array([0.2,0.2,0.2,0.2,0.2])

# Make analysis for equilibrium and dynamic limit
for method in ['eq','dyn']:
    # Set seed
    np.random.seed(1)
    
    # Predict energies, fill surface and get adsorbed sites and pairs
    CO_NO_energy_pairs, CO_ads, NO_ads, H_ads, CO_energies, NO_energies, H_energies, CO_bool, NO_bool, H_bool = get_sites(composition,P_CO=1,P_NO=1, metals=metals, method=method, eU=0, n=100, return_ads_energies=True)

    # Make 1D adsorption energy distribution plot
    fig, ax = plt.subplots(figsize=(3.4,2))
    
    # Make consistent bins
    Emin = np.min([CO_energies,NO_energies,H_energies])
    Emax = np.max(np.concatenate([CO_energies,NO_energies,H_energies]))
    range = (Emin,Emax)
    bins=100
    
    # Plot histogram of gross free energies
    ax.hist(CO_energies,histtype='step',label='CO',color=C_color,range=range,bins=bins,zorder=2)
    ax.hist(NO_energies,histtype='step',label='NO',color=N_color,range=range,bins=bins,zorder=1)
    # Plot histogram of net energies
    CO_counts, bins, patches = ax.hist(CO_ads,label='*CO',alpha=1,range=range,bins=bins,color=C_color,zorder=2)
    ax.hist(NO_ads,label='*NO',alpha=0.6,range=range,bins=bins,color=N_color,zorder=1)

    ax.set_xlabel('$\Delta G_{ads}$ [eV]',fontsize=9)
    ax.set_ylabel('Number of Sites',fontsize=9)
    ax.set_ylim(None,380)
    ax.set_xlim(-1.5,0.1)
    h,l = ax.get_legend_handles_labels()
    ax.legend(loc=2,handles=h[2:],labels=l[2:])

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))

    # Make zoomed inset for dynamic limit
    if method == 'dyn':
        axins = ax.inset_axes([0.03, 0.46, 0.2, 0.2])
        axins.hist(CO_ads[CO_ads<=-1],range=range,bins=bins,color=C_color)
        axins.set_xlim([-1.4,-1.0])
        axins.set_ylim([0,5.5])
        axins.set_yticks([])
        axins.set_xticklabels([])

        axins.xaxis.set_major_locator(MultipleLocator(0.5))
        axins.xaxis.set_minor_locator(MultipleLocator(0.1))
        axins.yaxis.set_minor_locator(MultipleLocator(1))
        
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="black",alpha=0.3,linestyle='--')

    plt.tight_layout()
    # plt.savefig(f'adsorption_dist_{method}.png',dpi=600)
    plt.savefig(f'adsorption_dist_{method}.svg',format='svg')

    #Correct to adsorption energies
    NO_energies-=G_corr['NO']
    CO_energies-=G_corr['CO']
    H_energies-=G_corr['H']

    #Correct to adsorption energies
    NO_ads-=G_corr['NO']
    CO_ads-=G_corr['CO']
    CO_NO_energy_pairs[:,0] -= G_corr['CO']
    CO_NO_energy_pairs[:,1] -= G_corr['NO']


    # Remove H blocked energies from gross distribution
    H_bool_pad = np.pad(H_bool,pad_width=((1,1),(1,1)),mode='wrap')
    CO_block_mask = H_bool_pad[1:-1,1:-1] + H_bool_pad[:-2,1:-1] + H_bool_pad[1:-1,:-2]
    NO_energies = NO_energies[np.invert(H_bool).flatten()]
    CO_energies = CO_energies[np.invert(CO_block_mask).flatten()]


    # make probability distribution plot
    fig,ax = plt.subplots(figsize=(4,4))
    ax.set_ylim(-1.455,-0.123)
    ax.set_xlim(-1.82,-0.0265)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # Plot distribution
    fig,ax,(hist_x,xedges),(hist_y,yedges), prob_density = probability_density_plot(CO_energies,NO_energies,return_hists=True,figure=(fig,ax),vmax=80)
    
    # Plot pairs on distribution
    ax.scatter(*CO_NO_energy_pairs.T,marker='.',c='b',edgecolors='k',label=f'CO-NO pairs: {len(CO_NO_energy_pairs)}',s=14,linewidths=0.6)
    ax.legend(loc=2, markerscale=2.,framealpha=0.4)

    # Plot single elements on distribution
    ax.scatter(CO_metal_energies[0],NO_metal_energies[0],color=metal_colors['Ag'],edgecolors='k')
    ax.text(CO_metal_energies[0],NO_metal_energies[0]+0.02,'Ag',va='bottom',ha='center',fontsize=7)
    
    ax.scatter(CO_metal_energies[1],NO_metal_energies[1],color=metal_colors['Au'],edgecolors='k')
    ax.text(CO_metal_energies[1],NO_metal_energies[1]+0.02,'Au',va='bottom',ha='center',fontsize=7)
    
    ax.scatter(CO_metal_energies[2],NO_metal_energies[2],color=metal_colors['Cu'],edgecolors='k',alpha=1,zorder=0)
    ax.text(-0.39,NO_metal_energies[2],'Cu',va='center',ha='left',fontsize=7)

    # Linear regression scaling parameters
    a = 1.88
    b = - 1.24
    # Plot dG=0 and H eq. lines
    ax.vlines(-G_corr['CO'], ylim[0],ylim[1],color='k',ls='--',alpha=0.6,lw=1)
    ax.hlines(-G_corr['NO'], xlim[0],xlim[1],color='k',ls='--',alpha=0.6,lw=1)
    ax.hlines(a*(-0.1)+b, xlim[0],xlim[1],color='b',ls='--',lw=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


    ax.set_xlabel("\u0394$E_{pred}^{*CO}$ [eV]")
    ax.set_ylabel("\u0394$E_{pred}^{*NO}$ [eV]")


    ax.xaxis.set_major_locator(MultipleLocator(0.4))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))


    fig.subplots_adjust(bottom=0.2,left=0.2,right=0.8,top=0.8)


    # plt.savefig(f'Energy_dist_{method}.png',dpi=600,bbox_inches='tight')
    plt.savefig(f'Energy_dist_{method}.pdf',dpi=1000,bbox_inches='tight')