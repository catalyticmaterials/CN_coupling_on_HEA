import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append('..')
from scripts.surface import adsorb_CO, adsorb_NO, block_indices
from scripts.methods import count_CO_NO_pairs

# Simulation parameters
n=10
kBT = 1
N_iter=5000

# Initiate lists
CO_coverage_mask = np.full((n, n),None)
NO_coverage_mask = np.full((n, n),None)
n_pairs_list = [0]

# Set seed
np.random.seed(42)

# Simulation
for iteration in tqdm(range(N_iter)):
    # Get random coordinate on grid
    i,j = np.random.randint(0,n,size=2)
    
    # Pick randomly between CO and NO
    if np.random.random() < 0.5: # pick NO
        if NO_coverage_mask[i,j]:
            # Move on if NO is already adsorbed at site
            pass

        elif NO_coverage_mask[i,j]==None:
            # Adsorb if free
            CO_coverage_mask, NO_coverage_mask = adsorb_NO(i,j,CO_coverage_mask,NO_coverage_mask,n)

        else:
            # If blocked calculate the difference in number of pairs when removing blocking adsorbates
            block_ind = block_indices(i,j,'fcc',n)
            CO_coverage_bool_new = np.copy(CO_coverage_bool)
            for ib,jb in block_ind:
                CO_coverage_bool_new[ib,jb] = False

            NO_coverage_bool_new = np.copy(NO_coverage_bool)
            NO_coverage_bool_new[i,j] = True

            # Get new number of pairs
            n_pairs_new = count_CO_NO_pairs(CO_coverage_bool_new,NO_coverage_bool_new)
            # Get difference
            n_pairs_diff = n_pairs_new - n_pairs
            # Accept new adsorption according to Boltzmann prob.
            if n_pairs_diff>0 or np.random.random() <= np.exp(n_pairs_diff/kBT):
                CO_coverage_mask,NO_coverage_mask = adsorb_NO(i,j,CO_coverage_mask,NO_coverage_mask,n)

    else: # Pick CO
        if CO_coverage_mask[i,j]:
            # Move on if CO is already adsorbed at site
            pass
        
        elif CO_coverage_mask[i,j]==None:
            # Adsorb if free
            CO_coverage_mask, NO_coverage_mask = adsorb_CO(i,j,CO_coverage_mask,NO_coverage_mask,n)

        else:
            # If blocked calculate the difference in number of pairs when removing blocking adsorbates
            block_ind = block_indices(i,j,'top',n)
            NO_coverage_bool_new = np.copy(NO_coverage_bool)
            for ib,jb in block_ind:
                NO_coverage_bool_new[ib,jb] = False

            CO_coverage_bool_new = np.copy(CO_coverage_bool)
            CO_coverage_bool_new[i,j] = True
            
            # Get new number of pairs
            n_pairs_new = count_CO_NO_pairs(CO_coverage_bool_new,NO_coverage_bool_new)
            # Get difference
            n_pairs_diff = n_pairs_new - n_pairs
            # Accept new adsorption according to Boltzmann prob.
            if n_pairs_diff>0 or np.random.random() <= np.exp(n_pairs_diff/kBT):
                CO_coverage_mask,NO_coverage_mask = adsorb_CO(i,j,CO_coverage_mask,NO_coverage_mask,n)




    # Convert coverage mask into all boolean values
    CO_coverage_bool = np.asanyarray(CO_coverage_mask,dtype=bool)
    NO_coverage_bool = np.asanyarray(NO_coverage_mask,dtype=bool)

    # Count number of pairs at iteration
    n_pairs = count_CO_NO_pairs(CO_coverage_bool,NO_coverage_bool)
    n_pairs_list.append(n_pairs)
    # Lower temperature by 1%
    kBT = kBT*0.99

print(kBT)
print(np.max(n_pairs_list))
print(np.sum(CO_coverage_bool),np.sum(NO_coverage_bool))

# Plot number of pairs per iteration
fig,ax = plt.subplots(figsize=(3,3))
ax.plot(n_pairs_list,c='k',label='simulation')
ax.set_xlabel('Iterations (units of 1000)')
ax.set_ylabel('Pairs per surface atoms')
ax.set_yticks([0,0.25,0.50,0.75])
ax.set_xticks([0,1000,2000,3000,4000,5000])
ax.set_xticklabels([0,1,2,3,4,5])
ax.hlines(0.75,*ax.get_xlim(),colors='k',alpha=0.6,ls='--',lw=1)
ax.set_xlim(0,5000)
plt.subplots_adjust(bottom=0.2,left=0.2)
plt.savefig('max_pairs_sim.png',dpi=600,bbox_inches='tight')




# Plot final grid
fig, ax = plt.subplots(figsize=(4,4))

# Grid ids of adsorbed CO
i_CO,j_CO = np.where(CO_coverage_bool)
for i,j in zip(i_CO,j_CO):
    # Make vecor from fcc basis vector
    xy = i*np.array([1,0]) + j*np.array([0.5,np.sqrt(3)/2])
    # Plot CO
    ax.scatter(*xy,c='k',zorder=2,s=70)
    ax.scatter(*xy,c='r',zorder=3,s=20)

# Grid ids of adsorbed CO
i_NO,j_NO = np.where(NO_coverage_bool)
for i,j in zip(i_NO,j_NO):
    # Make vecor from fcc basis vector
    xy = i*np.array([1,0]) + j*np.array([0.5,np.sqrt(3)/2]) + np.array([0.5,np.sqrt(3)/6])
    # Plot NO
    ax.scatter(*xy,c='b',zorder=2,s=70)
    ax.scatter(*xy,c='r',zorder=3,s=20)

# Plot grid positions
x_coordinates = np.zeros((n,n))
y_coordinates = np.zeros((n,n))
iarr,jarr = np.meshgrid(range(n),range(n))
for i,j in zip(iarr.flatten(),jarr.flatten()):
    xy = i*np.array([1,0]) + j*np.array([0.5,np.sqrt(3)/2]) 
    ax.scatter(*xy,facecolors='white',edgecolors='k',zorder=1,s=10)
    x_coordinates[i,j] = xy[0]
    y_coordinates[i,j] = xy[1]

# Plot gridlines
for i in range(n):
    ax.plot([x_coordinates[0,i],x_coordinates[-1,i]],[y_coordinates[0,i],y_coordinates[-1,i]],c='k',lw=1,zorder=0)
    ax.plot([x_coordinates[i,0],x_coordinates[i,-1]],[y_coordinates[i,0],y_coordinates[i,-1]],c='k',lw=1,zorder=0)

plt.axis('off')
ax.set_aspect('equal')
plt.savefig('max_pairs_result.png',dpi=600,bbox_inches='tight')