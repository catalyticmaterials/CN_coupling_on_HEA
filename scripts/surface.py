import numpy as np
import iteround
from joblib import load
import itertools as it
from time import time

# Define Boltzmann's constant
#kB = 1.380649e-4 / 1.602176634  # eV K-1 (exact)
kB=8.617333262 * 1e-5 #eV/K
kBT = kB*300 # eV

metals = ['Ag','Au', 'Cu', 'Pd','Pt']

# Define relative blocking vectors
block_fcc_vectors = np.array([[-1,0],[0,-1],[0,0]])
block_top_vectors = np.array([[0,0],[0,1],[1,0]])

def block_indices(i,j,ads_site,n=100):
    if ads_site=="top":
        block_vectors = np.array([i,j]) + block_fcc_vectors
    elif ads_site=="fcc":
        block_vectors = np.array([i,j]) + block_top_vectors
    block_vectors[block_vectors==n] = 0
    return block_vectors

def adsorb_H(i,j,H_coverage_mask,CO_coverage_mask,NO_coverage_mask,n=100):
    H_coverage_mask[i,j] = True
    
    i_b,j_b = block_indices(i,j,"fcc",n).T
    CO_coverage_mask[i_b,j_b] = False
    NO_coverage_mask[i,j] = False
    return H_coverage_mask,CO_coverage_mask,NO_coverage_mask

def adsorb_CO(i,j,CO_coverage_mask,NO_coverage_mask,n=100):
    CO_coverage_mask[i,j] = True
    #CO_energies = np.append(CO_energies,energy)
    
    #Block sites
    i_b,j_b = block_indices(i,j,"top",n).T
    NO_coverage_mask[i_b,j_b] = False
        #H_coverage_mask[i_b,j_b] = False
    return CO_coverage_mask,NO_coverage_mask

def adsorb_NO(i,j,CO_coverage_mask,NO_coverage_mask,n=100):
    NO_coverage_mask[i,j] = True
    #NO_energies = np.append(NO_energies,energy)
    
    #Block sites
    i_b,j_b = block_indices(i,j,"fcc",n).T
    CO_coverage_mask[i_b,j_b] = False
    #H_coverage_mask[i,j] = False
    return CO_coverage_mask,NO_coverage_mask

def initiate_surface(f,metals,size=(100,100),n_layers=3):
    
    #Number of atoms in surface
    n_surface_atoms = np.prod(size)
    #Total number of atoms
    n_atoms = n_surface_atoms*n_layers
    #number of each metal
    n_each_metal = f*n_atoms
    #Round to integer values while maintaining sum
    n_each_metal=iteround.saferound(n_each_metal, 0)
    
    assert(np.sum(n_each_metal)==n_atoms)

    #Make list of metals
    symbols=[]
    for i in range(len(metals)):
        symbols+=[metals[i]] * int(n_each_metal[i])
   
    #Shuffle the elements
    np.random.shuffle(symbols)
    
    #Make 3D grid as surface
    surface = np.reshape(symbols,(*size,n_layers))
    return surface

def fingerprint(grid_coords,surface_grid,adsorbate,metals):
    # Define relative ids of on-top neighbor atoms
    ontop_1a = np.array([(0,0,0)])
    ontop_1b = np.array([(0,-1,0), (-1,0,0), (-1,1,0), (0,1,0), (1,0,0), (1,-1,0)])
    ontop_2a = np.array([(-1,0,1), (0,-1,1), (0,0,1)])
    ontop_3b = np.array([(-1,-1,2), (-1,0,2), (0,-1,2)])
    
    # Define relative ids of fcc neighbor atoms
    fcc_1a = np.array([(0,1,0), (1,0,0), (1,1,0)]) 
    fcc_1b = np.array([(0,0,0), (0,2,0), (2,0,0)])
    #fcc_1c = np.array([(-1,1,0), (-1,2,0), (1,-1,0), (2,-1,0), (1,2,0), (2,1,0)])
    fcc_2a = np.array([(0,0,1), (0,1,1), (1,0,1)])
    #fcc_2b = np.array([(-1,1,1), (1,-1,1), (1,1,1)])
    
    if adsorbate=="CO" or adsorbate=="NO":
        ads_pos = ontop_1a
        zone_pos = [ontop_1b,ontop_2a,ontop_3b]
        n_atoms_site=1
        
    else:
        ads_pos = fcc_1a
        zone_pos = [fcc_1b,fcc_2a]
        n_atoms_site=3
    
    #Surface properties
    surface_size=surface_grid.shape
    n_metals=len(metals)
    
    # Get the unique on-top sites
    unique_sites = list(it.combinations_with_replacement(metals, n_atoms_site))
    #Number of unique sites
    n_sites=len(unique_sites)
    
    # Get element ids of neighbor atoms
    ads_ids = surface_grid[tuple(((ads_pos + grid_coords) % surface_size).T)]
    zone_ids = [surface_grid[tuple(((pos + grid_coords) % surface_size).T)] for pos in zone_pos]

    #Get site index of adsorption site
    site_idx = unique_sites.index(tuple(sorted(ads_ids)))

    #Get fingerprint
    fp_site = [0]*n_sites
    fp_site[site_idx] = 1
    fp_zones = [sum(zone == elem) for zone in zone_ids for elem in metals]
    fingerprint = fp_site + fp_zones
    return np.array(fingerprint)

def predict_energies(surface,adsorbate,metals):
    #Get shape of surface
    n_rows,n_cols,n_layers = surface.shape
    
    #load regressor
    reg = load(f'../train_model/{adsorbate}/{adsorbate}.joblib')
    
    #Initiate list to store adsorption energies of each site
    energies = np.empty(0)
    site_ids = np.empty((0,2))
    
    for row in range(n_rows):
        for col in range(n_cols):
            
            #site index
            site_id = np.array([[row,col]])
            
            #Get site fingerprint
            fp = fingerprint((row,col,0),surface,adsorbate,metals)
            
            #Predict energies
            if adsorbate=="H" or adsorbate=="NO_fcc":
                #Predict energy
                energy=reg.predict(fp)
            else:
                #split into ensemble and feature
                ensemble = fp[:5]
                feature = fp[5:]
                #Predict energy
                energy=reg.predict(tuple(ensemble),feature)
            
            #Append energy
            energies = np.append(energies,energy)
            site_ids = np.vstack((site_ids,site_id))
    return energies, site_ids


def fill_equilibrium(CO_data,NO_data,CO_coverage_mask,NO_coverage_mask,H_coverage_mask):
    # Make ids for CO and NO
    CO_ids=np.zeros((len(CO_data),1))
    NO_ids=np.ones((len(NO_data),1))

    # Include ids in data array
    CO_data = np.hstack((CO_data,CO_ids.reshape(-1,1)))
    NO_data = np.hstack((NO_data,NO_ids.reshape(-1,1)))

    # Stack arrays
    CO_NO_array = np.vstack((CO_data,NO_data))
    
    #Sort data by lowest energy
    CO_NO_array = CO_NO_array[CO_NO_array[:, 0].argsort()]

    # Initiate Coverage masks
    CO_coverage_mask = np.full((100, 100),None)
    NO_coverage_mask = np.full((100, 100),None)

    #Fill surface with NO and CO with blocking
    for (energy,i,j,idx) in CO_NO_array:
        if energy>0: break
        i,j,idx = int(i),int(j),int(idx)
        ij_vec = np.array([i,j])
            
        if idx==0:
            if CO_coverage_mask[i,j] is None:
                CO_coverage_mask,NO_coverage_mask=adsorb_CO(i, j, CO_coverage_mask, NO_coverage_mask)
        
        elif idx==1:
            if NO_coverage_mask[i,j] is None:
                CO_coverage_mask,NO_coverage_mask=adsorb_NO(i, j, CO_coverage_mask, NO_coverage_mask)

    # return coverage masks
    return CO_coverage_mask,NO_coverage_mask


def fill_dynamic(CO_data,NO_data,CO_coverage_mask,NO_coverage_mask,H_coverage_mask, P_CO,P_NO):
    
    # get partial preassures
    P = P_CO + P_NO
    p_CO = P_CO/P
    p_NO=P_NO/P

    while len(CO_data)>0 or len(NO_data)>0:
        r = np.random.rand()
        if r<=p_CO and len(CO_data)>0:
            ind = np.random.choice(len(CO_data))
            energy, i, j = CO_data[ind]
            i,j = int(i),int(j)
            CO_data = np.delete(CO_data, ind,axis=0)
            if CO_coverage_mask[i,j] is None:
                CO_coverage_mask,NO_coverage_mask=adsorb_CO(i, j, CO_coverage_mask, NO_coverage_mask)
                
        elif len(NO_data)>0:
            ind = np.random.choice(len(NO_data))
            energy, i, j = NO_data[ind]
            i,j = int(i),int(j)
            NO_data = np.delete(NO_data, ind,axis=0)
            if NO_coverage_mask[i,j] is None:
                CO_coverage_mask,NO_coverage_mask=adsorb_NO(i, j, CO_coverage_mask, NO_coverage_mask)
                

    # return coverage masks
    return CO_coverage_mask,NO_coverage_mask

def fill_mc(CO_data,NO_data,CO_coverage_mask,NO_coverage_mask,H_coverage_mask, P_CO,P_NO, n=100):
    
    # Get energies as grids
    CO_energy_grid = CO_data[:,0].reshape(n,n)
    NO_energy_grid = NO_data[:,0].reshape(n,n)

    # get partial preassures
    P = P_CO + P_NO
    p_CO = P_CO/P
    p_NO=P_NO/P

    no_ads_count = 0

    while no_ads_count<=1000:
        no_ads_count += 1
        r = np.random.rand()
        if r<= p_CO: # Adsorb CO
            # Pick a random site to adsorb
            ind = np.random.choice(len(CO_data))#,p=p)
            energy, i, j = CO_data[ind]
            # Check if it can adsorb on the given site
            if energy>0:
                continue
            # Make sure site indexes are int
            i,j = int(i),int(j)

            # Get indices of blocking sites
            block_vectors = block_indices(i, j, "top",n)
            
            # Check if the site is blocked by H
            H_blocked = np.any([H_coverage_mask[ib,jb] for (ib,jb) in block_vectors])

            # Continue to next iteration if the picked site is blocked by a hydrogen
            if H_blocked:
                continue

            # Adsorb if site is vacant
            elif CO_coverage_mask[i,j] is None:
                CO_coverage_mask,NO_coverage_mask=adsorb_CO(i, j, CO_coverage_mask, NO_coverage_mask,n)
                # Reset no direct adsorbtion counter
                no_ads_count = 0

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
                continue
            # Make sure site indexes are int
            i,j = int(i),int(j)
            
            # Get indices of blocking sites
            block_vectors = block_indices(i, j, "fcc",n)
            
            # Check if the site is blocked by H
            H_blocked = H_coverage_mask[i,j]==True
                        
            # Continue to next iteration if the picked site is blocked by a hydrogen
            if H_blocked:
                continue

            # Adsorb if site is vacant    
            elif NO_coverage_mask[i,j] is None:
                CO_coverage_mask,NO_coverage_mask=adsorb_NO(i, j, CO_coverage_mask, NO_coverage_mask,n)
                no_ads_count = 0

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
    # return coverage masks
    return CO_coverage_mask,NO_coverage_mask


def fill_surface(composition,P_CO,P_NO, metals, method, eU=0, n=100):

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

    

    if method=='eq':
        CO_coverage_mask,NO_coverage_mask = fill_equilibrium(CO_data,NO_data,CO_coverage_mask,NO_coverage_mask,H_coverage_mask)
    elif method=='dyn':
        CO_coverage_mask,NO_coverage_mask = fill_dynamic(CO_data,NO_data,CO_coverage_mask,NO_coverage_mask,H_coverage_mask, P_CO,P_NO)
    elif method=='mc':
        CO_coverage_mask,NO_coverage_mask = fill_mc(CO_data,NO_data,CO_coverage_mask,NO_coverage_mask,H_coverage_mask, P_CO,P_NO, n)


    # Convert coverage mask into all boolean values
    CO_coverage_bool = np.asanyarray(CO_coverage_mask,dtype=bool)
    NO_coverage_bool = np.asanyarray(NO_coverage_mask,dtype=bool)
    H_coverage_bool = np.asanyarray(H_coverage_mask,dtype=bool)

    # Turn into grids
    CO_energy_grid = CO_energies.reshape(n,n)
    NO_energy_grid = NO_energies.reshape(n,n)
    H_energy_grid = H_energies.reshape(n,n)

    # Return coverage masks and energy grids
    return (CO_coverage_bool, NO_coverage_bool, H_coverage_bool), (CO_energy_grid, NO_energy_grid, H_energy_grid)

# def fill_surface(CO_energies,CO_site_ids,NO_energies,NO_site_ids):
    
#     #Append index to data
#     CO_id = np.ones((len(CO_energies),1))
#     NO_id = np.ones((len(NO_energies),1))*2
#     CO = np.hstack((CO_energies.reshape(-1,1),CO_site_ids,CO_id))
#     NO = np.hstack((NO_energies.reshape(-1,1),CO_site_ids,NO_id))

#     #Combine data
#     data_array=np.vstack((CO,NO))
    
#     #Sort data by lowest energy
#     data_array = data_array[data_array[:, 0].argsort()]
    
#     #Prepare grids
#     fcc_grid = np.zeros((100,100))
#     top_grid = fcc_grid.copy()
    
    
#     block_fcc_vectors = np.array([[-1,0],[0,-1],[0,0]])
#     block_top_vectors = np.array([[-1,0],[-1,1],[1,0]])

    
#     for (energy,i,j,idx) in data_array:
#         if energy>0: break
#         i,j,idx = int(i),int(j),int(idx)
#         ij_vec = np.array([i,j])
    
            
#         if idx==1:
#             if top_grid[i,j]==0:
#                 top_grid[i,j] = energy
#                 #CO_energies = np.append(CO_energies,energy)
                
#                 #Block sites
#                 ij_block_vectors = ij_vec + block_fcc_vectors
#                 for (i_b,j_b) in ij_block_vectors:
#                     fcc_grid[i_b,j_b] = 1
#             else:
#                 continue
        
#         elif idx==2:
#             if fcc_grid[i,j]==0:
#                 fcc_grid[i,j] = energy
#                 #NO_energies = np.append(NO_energies,energy)
                
#                 #Block sites
#                 ij_block_vectors = ij_vec + block_top_vectors
#                 for (i_b,j_b) in ij_block_vectors:
#                     if i_b == 100:
#                         i_b=0
#                     elif j_b == 100:
#                         j_b=0
#                     top_grid[i_b,j_b] = 1
    
#     return top_grid,fcc_grid



# def characterize_sites(fcc_grid,top_grid):
#     CO_ids = np.array(np.nonzero(top_grid<0)).T
#     CO_NO_energy_pair = np.empty((0,2))
    
    
#     #pad grids
#     fcc_grid = np.pad(fcc_grid,pad_width=1,mode="wrap")
#     top_grid = np.pad(top_grid,pad_width=1,mode="wrap")
#     CO_ids+=1
    
#     NO_ads_mask = fcc_grid < 0
    
#     #Get CO-NO pairs of catalytic neighboring sites
#     for (i,j) in CO_ids:
#         if fcc_grid[i-1,j-1] < 0:
#             E_CO = E_CO = top_grid[i,j]#CO_energies[i*100+j]
#             E_NO = fcc_grid[i-1,j-1] #NO_energies[(i-1)*100+(j-1)]
#             NO_ads_mask[i-1,j-1] = False
#             CO_NO_energy_pair = np.vstack((CO_NO_energy_pair,np.array([[E_CO,E_NO]])))
#         if fcc_grid[i-1,j+1] < 0:
#             E_CO = E_CO = top_grid[i,j]#CO_energies[i*100+j]
#             E_NO = fcc_grid[i-1,j+1] #NO_energies[(i-1)*100+(j+1)]
#             NO_ads_mask[i-1,j+1] = False
#             CO_NO_energy_pair = np.vstack((CO_NO_energy_pair,np.array([[E_CO,E_NO]])))
#         if fcc_grid[i+1,j-1] < 0:
#             E_CO = top_grid[i,j] #CO_energies[i*100+j]
#             E_NO = fcc_grid[i+1,j-1] #NO_energies[(i+1)*100+(j-1)]
#             NO_ads_mask[i+1,j-1] = False
#             CO_NO_energy_pair = np.vstack((CO_NO_energy_pair,np.array([[E_CO,E_NO]])))
            
            
#         fcc_energies=fcc_grid[1:-1,1:-1].flatten()
#         NO_ads_mask = NO_ads_mask[1:-1,1:-1].flatten()
        
#         #Remaining fcc sites which are not paired with CO
#         NH3_site_energies = fcc_energies[NO_ads_mask]
        
#         return CO_NO_energy_pair, NH3_site_energies
    