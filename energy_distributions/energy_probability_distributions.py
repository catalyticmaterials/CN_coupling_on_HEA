import numpy as np
import matplotlib.pyplot as plt
from density_plot_functions import probability_density_plot
from matplotlib.ticker import MultipleLocator
from ase.db import connect

import sys
sys.path.append('..')
from scripts.surface import predict_energies, adsorb_H, initiate_surface, kBT
from scripts import metals, metal_colors

P_NO=1
P_CO=1
n=100
eU=0


# Get adsorption energies on single element
db_path = "../databases"
with connect(f'{db_path}/single_element_slabs_out.db') as db_slab,\
    connect(f'{db_path}/single_element_H_out.db') as db_H,\
        connect(f'{db_path}/single_element_CO_out.db') as db_CO,\
            connect(f'{db_path}/single_element_NO_fcc_out.db') as db_NO:
        
        
        E_ref_H = db_H.get(metal='Cu').energy - db_slab.get(metal='Cu').energy -0.1 + 0.1
        E_ref_CO = db_CO.get(metal='Cu').energy - db_slab.get(metal='Cu').energy +0.18 + 0.4
        E_ref_NO = db_NO.get(metal='Cu').energy - db_slab.get(metal='Cu').energy +0.45 + 0.71 
                
        H_metal_energies=[db_H.get(metal=m).energy - db_slab.get(metal=m).energy - E_ref_H for m in metals]
        CO_metal_energies=[db_CO.get(metal=m).energy - db_slab.get(metal=m).energy - E_ref_CO for m in metals]
        NO_metal_energies=[db_NO.get(metal=m).energy - db_slab.get(metal=m).energy - E_ref_NO for m in metals]




compositions = [np.array([0.2,0.2,0.2,0.2,0.2]),np.array([0.1,0.1,0.6,0.1,0.1]),np.array([0.0,0.5,0.5,0.0,0.0])]


for composition in compositions:
    alloy_list = [metals[i] + str(int(composition[i]*100)) for i in range(len(metals)) if composition[i]>0]
    alloy = "".join(alloy_list)
    
    # Set the same seed for every composition
    np.random.seed(1)

    #Initiate surface
    surface = initiate_surface(composition,metals,size=(n,n))
    
    #Predict energies of every site
    CO_energies,CO_site_ids=predict_energies(surface, "CO", metals)
    NO_energies,NO_site_ids=predict_energies(surface, "NO_fcc", metals)
    H_energies,H_site_ids = predict_energies(surface,"H",metals)

    #Adjust free energy by chemical potential from preassure
    CO_energies += kBT * np.log(1/P_CO)
    NO_energies += kBT * np.log(1/P_NO)

    # Combine energies and ids
    CO_data = np.hstack((CO_energies.reshape(-1,1),CO_site_ids))
    NO_data = np.hstack((NO_energies.reshape(-1,1),NO_site_ids))
    H_data = np.hstack((H_energies.reshape(-1,1),H_site_ids))
    
    # Initiate coverage masks
    CO_coverage_mask = np.full((n, n),None)
    NO_coverage_mask = np.full((n, n),None)
    H_coverage_mask = np.full((n, n),None)
    
    # UPD H: Fill surface with H where adsorption energy is negative
    for (energy, i, j) in H_data:
        i,j=int(i),int(j)
        if energy<=(-eU):
            H_coverage_mask,CO_coverage_mask,NO_coverage_mask = adsorb_H(i,j,H_coverage_mask, CO_coverage_mask, NO_coverage_mask,n=n)


    CO_mask = np.invert(CO_coverage_mask == False).flatten()
    NO_mask = np.invert(NO_coverage_mask == False).flatten()

    CO_energies = CO_energies[CO_mask]
    NO_energies = NO_energies[NO_mask]
    
    CO_energies += -0.4
    NO_energies += -0.71


    fig,ax = plt.subplots(figsize=(5,5))
    fig.subplots_adjust(wspace=0.5)
    ax.set_ylim(-1.5,-0.34)
    ax.set_xlim(-1.8,-0.1)
    

    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    

    fig,ax,(hist_x,xedges),(hist_y,yedges), prob_density = probability_density_plot(CO_energies,NO_energies,return_hists=True,figure=(fig,ax),vmax=80)


    a = 1.88
    b = - 1.24




    ax.vlines(-0.4, ylim[0],ylim[1],color='k',ls='--',alpha=0.6)
    ax.hlines(-0.71, xlim[0],xlim[1],color='k',ls='--',alpha=0.6)
    ax.hlines(a*(-0.1)+b, xlim[0],xlim[1],color='b',ls='--')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


    ax.set_xlabel("\u0394$E_{DFT}^{*CO}$ [eV]")
    ax.set_ylabel("\u0394$E_{DFT}^{*NO}$ [eV]")


    ax.xaxis.set_major_locator(MultipleLocator(0.4))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))


    if np.all(composition==compositions[2]):
         pass
    else:
        ax.scatter(CO_metal_energies[0],NO_metal_energies[0],color=metal_colors['Ag'],edgecolors='k',label='Ag')
        ax.text(CO_metal_energies[0],NO_metal_energies[0]+0.02,'Ag',va='bottom',ha='center',fontsize=8)
    
    ax.scatter(CO_metal_energies[1],NO_metal_energies[1],color=metal_colors['Au'],edgecolors='k',label='Au')
    ax.text(CO_metal_energies[1],NO_metal_energies[1]+0.02,'Au',va='bottom',ha='center',fontsize=8)
    
    ax.scatter(CO_metal_energies[2],NO_metal_energies[2],color=metal_colors['Cu'],edgecolors='k',label='Cu')
    ax.text(CO_metal_energies[2],NO_metal_energies[2]+0.02,'Cu',va='bottom',ha='center',fontsize=8)


    #ax.legend(loc=2)

    


    probability = np.sum(NO_energies<=(-0.71))/10000 * np.sum(CO_energies<=(-0.4))/10000

    print(probability)


    # x_centers = (xedges[1:] + xedges[:-1])/2
    # y_centers = (yedges[1:] + yedges[:-1])/2

    # x_centers_id=np.where(x_centers<=-0.4)[0]
    # y_centers_id=np.where(y_centers<=-0.71)[0]

    #prob2 = np.sum(prob_density.T[y_centers_id[0]:y_centers_id[-1]+1,x_centers_id[0]:x_centers_id[-1]+1]) * (x_centers[x_centers_id[1]]-x_centers[x_centers_id[0]]) * (y_centers[y_centers_id[1]]-y_centers[y_centers_id[0]]) 

    #print(prob2)
    alloy_sub_list = [metals[i] + '$_{' + str(int(composition[i]*100)) + '}$' for i in range(len(metals)) if composition[i]>0]
    alloy_sub = ''.join(alloy_sub_list)
    props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    ax.text(xlim[0]+0.05,ylim[1]-0.05,alloy_sub,ha='left',va='top',color='k',bbox=props)

    ax.text(xlim[0]+0.02,-0.73,f'$P_{{area}}$ = {probability:.4f}',ha='left',va='top',color='tab:blue')
    ax.fill_between([xlim[0],-0.4],ylim[0],-0.71,alpha=0.3,hatch='///',color='none',linewidth=0,edgecolor='tab:blue')
    

    #ax.set_aspect('equal')
    #plt.tight_layout()
    #plt.show()
    # plt.savefig('test.png',dpi=400)
    # break
    fig.subplots_adjust(bottom=0.2,left=0.2,right=0.8,top=0.8)
   
    plt.savefig(f'{alloy}/{alloy}_CO_NO_energy_distribution.png',dpi=600,bbox_inches='tight')
    break


    


    


