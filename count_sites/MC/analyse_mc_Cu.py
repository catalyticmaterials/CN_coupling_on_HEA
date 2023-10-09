import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys
from tqdm import tqdm
sys.path.append('..')
from scripts.surface import adsorb_CO, adsorb_NO, adsorb_H, block_indices, predict_energies, initiate_surface, kBT, metals

# set general parameters
P_CO=1
composition = np.array([0.,0.,1.0,0.,0.])
n=100
eU=0

max_ads_rate = []
break_iterations = []
fig,ax = plt.subplots(figsize=(8,3))
ax2=ax.twinx()
for P_NO in [1,0.1]:
    np.random.seed(42)
    # initiate lists
    total_ads_energies = []
    ads_rate = []
    iterations = []

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
            H_coverage_mask,CO_coverage_mask,NO_coverage_mask = adsorb_H(i,j,H_coverage_mask, CO_coverage_mask, NO_coverage_mask)


    # Get energies as grids
    CO_energy_grid = CO_data[:,0].reshape(n,n)
    NO_energy_grid = NO_data[:,0].reshape(n,n)
    H_energy_grid = H_data[:,0].reshape(n,n)


    # Get energy of adsorped sites
    H_ads = H_energy_grid[np.asanyarray(H_coverage_mask,dtype=bool)]
    # Get the initial total adsorption energy
    initial_ads_energy = np.sum(H_ads)

    # get partial preassures
    P = P_CO + P_NO
    p_CO = P_CO/P
    p_NO=P_NO/P

    ads_count = 0
    break_bool = False

    for iteration in tqdm(range(1,int(2.5e+5)+1)):
        r = np.random.rand()
        if r<= p_CO: # Adsorb CO
            # Pick a random site to adsorb
            ind = np.random.choice(len(CO_data))#,p=p)
            energy, i, j = CO_data[ind]
            # Check if it can adsorb on the given site
            if energy>0:
                pass
            else:
                # Make sure site indexes are int
                i,j = int(i),int(j)

                # Get indices of blocking sites
                block_vectors = block_indices(i, j, "top",n)
                
                # Check if the site is blocked by H
                H_blocked = np.any([H_coverage_mask[ib,jb] for (ib,jb) in block_vectors])

                # Continue to next iteration if the picked site is blocked by a hydrogen
                if H_blocked:
                    pass

                # Adsorb if site is vacant
                elif CO_coverage_mask[i,j] is None:
                    CO_coverage_mask,NO_coverage_mask=adsorb_CO(i, j, CO_coverage_mask, NO_coverage_mask,n)
                    # Add one to direct ads counter
                    ads_count += 1

                # Check if blocked by NO, calculate energy difference from exchange
                elif CO_coverage_mask[i,j]==False:
                    E=0
                    NO_blocking_ind = np.empty((0,2),dtype="int")
                    # Go through each blocking site to calculate the current energy
                    for (i_b,j_b) in block_vectors:
                        if NO_coverage_mask[i_b,j_b]:
                            E += NO_energy_grid[i_b,j_b]
                            NO_blocking_ind = np.vstack((NO_blocking_ind,np.array([[i_b,j_b]])))
                    
                    # Calculate the potential change in energy
                    Delta_E = CO_energy_grid[i,j] - E
                    # make exchange with Boltzmann probability
                    if Delta_E<0 or np.random.rand() <= np.exp(-Delta_E/kBT):
                        # Loop through removed NO to unblock CO sites
                        for i_b,j_b in NO_blocking_ind:
                            i_ub,j_ub = block_indices(i_b, j_b, "fcc",n).T
                            # Check if site is also blocked by H
                            not_H_blocked = np.array([np.any(H_coverage_mask[block_indices(ib_, jb_, "top",n).T])==False for ib_,jb_ in zip(i_ub,j_ub)])
                            # Unblock CO if not blocked by H
                            CO_coverage_mask[i_ub,j_ub][not_H_blocked]=None
                        # Adsorb CO
                        CO_coverage_mask,NO_coverage_mask=adsorb_CO(i, j, CO_coverage_mask, NO_coverage_mask,n)
                    
        else: # Adsorb NO
            # Pick random site to adsorb        
            ind = np.random.choice(len(NO_data))#,p=p)
            energy, i, j = NO_data[ind]
            # Check if it can adsorb on the given site
            if energy>0:
                pass
            else:
                    
                # Make sure site indexes are int
                i,j = int(i),int(j)
                
                # Get indices of blocking sites
                block_vectors = block_indices(i, j, "fcc",n)
                
                # Check if the site is blocked by H
                H_blocked = H_coverage_mask[i,j]==True
                            
                # Continue to next iteration if the picked site is blocked by a hydrogen
                if H_blocked:
                    pass

                # Adsorb if site is vacant    
                elif NO_coverage_mask[i,j] is None:
                    CO_coverage_mask,NO_coverage_mask=adsorb_NO(i, j, CO_coverage_mask, NO_coverage_mask,n)
                    # Add one to direct ads counter
                    ads_count += 1

                # Check if blocked by NO, calculate energy difference from exchange    
                elif NO_coverage_mask[i,j]==False:
                        E=0
                        CO_blocking_ind = np.empty((0,2),dtype="int")
                        # Go through each blocking site to calculate the current energy
                        for (i_b,j_b) in block_vectors:
                            if CO_coverage_mask[i_b,j_b]:
                                E += CO_energy_grid[i_b,j_b]
                                CO_blocking_ind = np.vstack((CO_blocking_ind,np.array([[i_b,j_b]])))
                        # Calculate the potential change in energy        
                        Delta_E = NO_energy_grid[i,j] - E
                        # make exchange with Boltzmann probability
                        if Delta_E<0 or np.random.rand() <= np.exp(-Delta_E/kBT):
                            # Loop through removed NO to unblock CO sites
                            for i_b,j_b in CO_blocking_ind:
                                i_ub,j_ub = block_indices(i_b, j_b, "top",n).T
                                # Check if site is also blocked by H
                                not_H_blocked = H_coverage_mask[i_ub,j_ub]==False
                                # Unblock NO if not blocked by H
                                NO_coverage_mask[i_ub,j_ub][not_H_blocked]=None
                            # Adsorb NO
                            CO_coverage_mask,NO_coverage_mask=adsorb_NO(i, j, CO_coverage_mask, NO_coverage_mask,n)

        if iteration%1000==0:
            # Get energy of adsorped sites
            CO_ads = CO_energy_grid[np.asanyarray(CO_coverage_mask,dtype=bool)]
            NO_ads = NO_energy_grid[np.asanyarray(NO_coverage_mask,dtype=bool)]
            H_ads = H_energy_grid[np.asanyarray(H_coverage_mask,dtype=bool)]
            # Get the current total adsorption energy and append
            current_ads_energy = np.sum(CO_ads)+np.sum(NO_ads)+np.sum(H_ads)
            total_ads_energies.append(current_ads_energy)
            iterations.append(iteration)
            # append the number of direct adsorption in the last 1000 iterations
            ads_rate.append(ads_count/1000)

            # record point of break as the first time there is no direct adsorptions
            if ads_count==0 and break_bool==False:
                break_iterations.append(iteration)
                break_bool=True
            else:
                # reset direct adsorption counter
                ads_count = 0

    max_ads_rate.append(np.max(ads_rate))

    #fig,ax = plt.subplots(figsize=(8,3))
    if P_NO==1:
        l1,=ax.plot(np.insert(iterations,0,0)/1000,np.insert(total_ads_energies,0,initial_ads_energy)/1000,c="orangered",label="Total ads. energy, P$_{NO}=1$")
        # ax2=ax.twinx()
        l2,=ax2.plot(np.array(iterations)/1000,ads_rate,c="teal",label="Adsorption Rate, P$_{NO}=1$")

    else:
        l3,=ax.plot(np.insert(iterations,0,0)/1000,np.insert(total_ads_energies,0,initial_ads_energy)/1000,c="orangered",label="Total ads. energy, P$_{NO}=0.1$",alpha=0.5)

        l4,=ax2.plot(np.array(iterations)/1000,ads_rate,c="teal",label="Adsorption Rate, P$_{NO}=0.1$",alpha=0.5)

ax.set_xlim(0,250)
ax.set_ylim(None,initial_ads_energy/1000)
ax2.set_ylim(None,np.max(max_ads_rate))

ax.set_xlabel("Number of Iterations [in units of 1000]")
ax.set_ylabel("Energy (keV)")
ax2.set_ylabel("Direct Adsorption Rate")

ylim=ax.get_ylim()
ax.vlines(break_iterations[0]/1000,*ylim,colors='k',ls='--',alpha=0.8)
ax.vlines(break_iterations[1]/1000,*ylim,colors='k',ls='--',alpha=0.4)


ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.xaxis.set_major_locator(MultipleLocator(25))


ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax2.yaxis.set_major_locator(MultipleLocator(0.2))
       
ax.yaxis.set_minor_locator(MultipleLocator(0.1))    
ax2.yaxis.set_minor_locator(MultipleLocator(0.1))


text="Cu$_{100}$"
ax.legend(handles=[l1,l2,l3,l4],loc=1) 
   
ax.text(25,ylim[0] + 0.97*(ylim[1]-ylim[0]) , s=text,va='top',ha='left')

plt.tight_layout()
plt.savefig(f'MC/MC_Cu.png',dpi=600)

