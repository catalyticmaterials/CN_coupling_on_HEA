import numpy as np
from .surface import predict_energies, initiate_surface, fill_surface
from time import time



def count_CO_NO_pairs(CO_coverage_bool, NO_coverage_bool):
    #Pad grids
    NO_coverage_mask_pad = np.pad(NO_coverage_bool,pad_width=((1,1),(1,1)),mode="wrap")

    # Get pairs in all three directions
    pairs1_bool=CO_coverage_bool * NO_coverage_mask_pad[:-2,:-2]
    pairs2_bool=CO_coverage_bool * NO_coverage_mask_pad[2:,:-2]
    pairs3_bool=CO_coverage_bool * NO_coverage_mask_pad[:-2,2:]
    
    # Total number of pairs
    n_pairs = np.sum(pairs1_bool) + np.sum(pairs2_bool) + np.sum(pairs3_bool)
    
    # Number of atoms in the surface layer
    surface_atoms = np.multiply(*CO_coverage_bool.shape)                 
    
    # Get active sites as the number of pairs pr. surface atom
    active_sites = n_pairs/surface_atoms
    
    return active_sites


def CO_NO_pairs_energy(CO_coverage_bool, NO_coverage_bool,CO_energy_grid, NO_energy_grid):
    
    # Pad NO grids
    NO_coverage_mask_pad = np.pad(NO_coverage_bool,pad_width=((1,1),(1,1)),mode="wrap")
    NO_energy_grid_pad = np.pad(NO_energy_grid,pad_width=((1,1),(1,1)),mode="wrap")

    # Get pairs in all three directions
    pairs1_bool=CO_coverage_bool * NO_coverage_mask_pad[:-2,:-2]
    pairs2_bool=CO_coverage_bool * NO_coverage_mask_pad[2:,:-2]
    pairs3_bool=CO_coverage_bool * NO_coverage_mask_pad[:-2,2:]

    # Get the adsorption energies of the adsorbates that are in pairs
    energy_pairs_CO = np.concatenate(([CO_energy_grid[pairs1_bool],CO_energy_grid[pairs2_bool],CO_energy_grid[pairs3_bool]]))
    energy_pairs_NO = np.concatenate(([NO_energy_grid_pad[:-2,:-2][pairs1_bool],NO_energy_grid_pad[2:,:-2][pairs2_bool],NO_energy_grid_pad[:-2,2:][pairs3_bool]]))

    # Collect the adsorption energies in its pairs
    CO_NO_energy_pairs = np.vstack((energy_pairs_CO,energy_pairs_NO)).T

    return CO_NO_energy_pairs

def count_sites(composition,P_CO,P_NO, metals, method, eU=0, n=100, return_ads_energies=False):
    t_start = time()

    # Simulate surface
    (CO_coverage_bool, NO_coverage_bool, H_coverage_bool), (CO_energy_grid, NO_energy_grid, H_energy_grid) = fill_surface(composition,P_CO,P_NO, metals, method, eU=eU, n=n)

    # count sites
    active_sites = count_CO_NO_pairs(CO_coverage_bool, NO_coverage_bool)

    print(str(np.around(composition,decimals=4)),"fractional active sites:",active_sites,"evaluation time (s):",time()-t_start)

    if return_ads_energies:
        CO_ads = CO_energy_grid[CO_coverage_bool]
        NO_ads = NO_energy_grid[NO_coverage_bool]
        H_ads = H_energy_grid[H_coverage_bool]
        return active_sites, CO_ads, NO_ads, H_ads
    else:
        return active_sites

def get_sites(composition,P_CO,P_NO, metals, method, eU=0, n=100, return_ads_energies=False):
    #t_start = time()

    # Simulate surface
    (CO_coverage_bool, NO_coverage_bool, H_coverage_bool), (CO_energy_grid, NO_energy_grid, H_energy_grid) = fill_surface(composition,P_CO,P_NO, metals, method, eU=eU, n=n)

    # count sites
    CO_NO_energy_pairs = CO_NO_pairs_energy(CO_coverage_bool, NO_coverage_bool,CO_energy_grid, NO_energy_grid)

    #print(str(composition),"fractional active sites:",active_sites,"evaluation time (s):",time()-t_start)

    if return_ads_energies:
        CO_ads = CO_energy_grid[CO_coverage_bool]
        NO_ads = NO_energy_grid[NO_coverage_bool]
        H_ads = H_energy_grid[H_coverage_bool]
        return CO_NO_energy_pairs, CO_ads, NO_ads, H_ads, CO_energy_grid.flatten(), NO_energy_grid.flatten(), H_energy_grid.flatten(), CO_coverage_bool, NO_coverage_bool, H_coverage_bool
    else:
        return CO_NO_energy_pairs



def count_selectivity(composition,P_CO,P_NO, metals, method, eU=0, n=100):
    # Simulate surface
    (CO_coverage_bool, NO_coverage_bool, H_coverage_bool), (CO_energy_grid, NO_energy_grid, H_energy_grid) = fill_surface(composition,P_CO,P_NO, metals, method, eU=eU, n=n)

    # Number of adsorped H
    N_H_ads = np.sum(H_coverage_bool)

    # count CO-NO pairs
    # Pad grid
    # NO_coverage_mask_pad = np.pad(NO_coverage_bool,pad_width=((1,1),(1,1)),mode="wrap")
    # #N_CO_NO_pairs = count_CO_NO_pairs(CO_coverage_bool, NO_coverage_bool)
    # # Get pairs in all three directions
    # pairs1_bool=CO_coverage_bool * NO_coverage_mask_pad[:-2,:-2]
    # pairs2_bool=CO_coverage_bool * NO_coverage_mask_pad[2:,:-2]
    # pairs3_bool=CO_coverage_bool * NO_coverage_mask_pad[:-2,2:]
    
    # Total number of pairs
    # n_pairs = np.sum(pairs1_bool) + np.sum(pairs2_bool) + np.sum(pairs3_bool)
    
    # Number of atoms in the surface layer
    surface_atoms = np.multiply(*CO_coverage_bool.shape)                 
    
    # Get active sites as the number of pairs pr. surface atom
    # N_CO_NO_pairs = n_pairs#/surface_atoms
    
    # N_CN_NO = np.sum(pairs1_bool+pairs2_bool+pairs3_bool)#/surface_atoms

    # Get pairs in all three directions
    #non_pairs1_bool=np.invert(CO_coverage_bool) * NO_coverage_mask_pad[:-2,:-2]
    #non_pairs2_bool=np.invert(CO_coverage_bool) * NO_coverage_mask_pad[2:,:-2]
    #non_pairs3_bool=np.invert(CO_coverage_bool) * NO_coverage_mask_pad[:-2,2:]

    CO_coverage_mask_pad = np.pad(CO_coverage_bool,pad_width=((1,1),(1,1)),mode="wrap")
    # non_pairs1_bool=NO_coverage_bool * np.invert(CO_coverage_mask_pad[:-2,2:])
    # non_pairs2_bool=NO_coverage_bool * np.invert(CO_coverage_mask_pad[2:,:-2])
    # non_pairs3_bool=NO_coverage_bool * np.invert(CO_coverage_mask_pad[2:,2:])
    # print(non_pairs1_bool)
    # print(non_pairs2_bool)
    # print(non_pairs3_bool)

    # N_NH3_NO = np.sum(non_pairs1_bool*non_pairs2_bool*non_pairs3_bool)


    
    pairs1_bool=NO_coverage_bool * CO_coverage_mask_pad[:-2,:-2]
    pairs2_bool=NO_coverage_bool * CO_coverage_mask_pad[2:,:-2]
    pairs3_bool=NO_coverage_bool * CO_coverage_mask_pad[:-2,2:]

    N_CN_NO = np.sum(pairs1_bool+pairs2_bool+pairs3_bool)

    N_CO_NO_pairs = (np.sum(pairs1_bool) + np.sum(pairs2_bool) + np.sum(pairs3_bool))
    
    N_NH3_NO = np.sum(NO_coverage_bool) - N_CN_NO
    
    # print(np.sum(NO_coverage_bool))

    # print(CO_coverage_bool)
    # print(NO_coverage_bool)
    # print(H_coverage_bool)
    return N_H_ads/surface_atoms, N_NH3_NO/surface_atoms, N_CN_NO/surface_atoms, N_CO_NO_pairs/surface_atoms




# Define Boltzmann's constant
#kB = 1.380649e-4 / 1.602176634  # eV K-1 (exact)
kB=8.617333262 * 1e-5 #eV/K
kBT = kB*300 # eV

metals = ['Ag','Au', 'Cu', 'Pd','Pt']

# Define relative blocking vectors
block_fcc_vectors = np.array([[-1,0],[0,-1],[0,0]])
block_top_vectors = np.array([[0,0],[0,1],[1,0]])

def block_indices(i,j,ads_site):
    if ads_site=="top":
        block_vectors = np.array([i,j]) + block_fcc_vectors
    elif ads_site=="fcc":
        block_vectors = np.array([i,j]) + block_top_vectors
    block_vectors[block_vectors==100] = 0
    return block_vectors

def adsorb_H(i,j,H_coverage_mask,CO_coverage_mask,NO_coverage_mask):
    H_coverage_mask[i,j] = True
    
    i_b,j_b = block_indices(i,j,"fcc").T
    CO_coverage_mask[i_b,j_b] = False
    NO_coverage_mask[i,j] = False
    return H_coverage_mask,CO_coverage_mask,NO_coverage_mask

def adsorb_CO(i,j,CO_coverage_mask,NO_coverage_mask):
    CO_coverage_mask[i,j] = True
    #CO_energies = np.append(CO_energies,energy)
    
    #Block sites
    i_b,j_b = block_indices(i,j,"top").T
    NO_coverage_mask[i_b,j_b] = False
        #H_coverage_mask[i_b,j_b] = False
    return CO_coverage_mask,NO_coverage_mask

def adsorb_NO(i,j,CO_coverage_mask,NO_coverage_mask):
    NO_coverage_mask[i,j] = True
    #NO_energies = np.append(NO_energies,energy)
    
    #Block sites
    i_b,j_b = block_indices(i,j,"fcc").T
    CO_coverage_mask[i_b,j_b] = False
    #H_coverage_mask[i,j] = False
    return CO_coverage_mask,NO_coverage_mask


def count_sites_equilibtrium(composition,P_CO,P_NO,eU=0,H_UPD=True,return_ads=False):
    t_start = time()
    #Initiate surface
    surface = initiate_surface(composition,metals)
    
    #Predict energies of every site
    CO_energies,CO_site_ids=predict_energies(surface, "CO", metals)
    NO_energies,NO_site_ids=predict_energies(surface, "NO_fcc", metals)
    H_energies,H_site_ids = predict_energies(surface,"H",metals)
    
    
    
    #Adjust free energy by chemical potential from preassure
    CO_energies += kBT * np.log(1/P_CO)
    NO_energies += kBT * np.log(1/P_NO)
    
    #Turn into grids
    CO_energy_grid = CO_energies.reshape(100,100)
    NO_energy_grid = NO_energies.reshape(100,100)
    H_energy_grid = H_energies.reshape(100,100)
    
    
    
    CO_ids=np.ones(len(CO_energies))*1
    NO_ids=np.ones(len(NO_energies))*2
    #H_ids=np.ones(len(H_energies))*3
    
    
    CO_data = np.hstack((CO_energies.reshape(-1,1),CO_site_ids,CO_ids.reshape(-1,1)))
    NO_data = np.hstack((NO_energies.reshape(-1,1),NO_site_ids,NO_ids.reshape(-1,1)))
    H_data = np.hstack((H_energies.reshape(-1,1),H_site_ids))
    
    
    CO_NO_array = np.vstack((CO_data,NO_data))
    
    #Sort data by lowest energy
    CO_NO_array = CO_NO_array[CO_NO_array[:, 0].argsort()]
    
    #Coverage masks
    CO_coverage_mask = np.full((100, 100),None)
    NO_coverage_mask = np.full((100, 100),None)
    H_coverage_mask = np.full((100, 100),None)
    
    
    
    if H_UPD:
        #fill surface with H where adsorption energy is negative
        for (energy, i, j) in H_data:
            i,j=int(i),int(j)
            if energy<=(-eU):
                H_coverage_mask,CO_coverage_mask,NO_coverage_mask = adsorb_H(i,j,H_coverage_mask, CO_coverage_mask, NO_coverage_mask)
    
    
    #Fill surface with NO and CO with blocking
    for (energy,i,j,idx) in CO_NO_array:
        if energy>0: break
        i,j,idx = int(i),int(j),int(idx)
        ij_vec = np.array([i,j])
            
        if idx==1:
            if CO_coverage_mask[i,j] is None:
                CO_coverage_mask,NO_coverage_mask=adsorb_CO(i, j, CO_coverage_mask, NO_coverage_mask)
        
        elif idx==2:
            if NO_coverage_mask[i,j] is None:
                CO_coverage_mask,NO_coverage_mask=adsorb_NO(i, j, CO_coverage_mask, NO_coverage_mask)
            

    CO_coverage_bool = np.asanyarray(CO_coverage_mask,dtype=bool)
    NO_coverage_bool = np.asanyarray(NO_coverage_mask,dtype=bool)
    #H_coverage_mask = np.asanyarray(H_coverage_mask,dtype=bool)
    
    #Pad grids
    NO_coverage_mask_pad = np.pad(NO_coverage_bool,pad_width=((1,1),(1,1)),mode="wrap")
    #NO_energy_grid_pad = np.pad(NO_energy_grid,pad_width=((1,1),(1,1)),mode="wrap")


    pairs1_bool=CO_coverage_bool * NO_coverage_mask_pad[:-2,:-2]
    pairs2_bool=CO_coverage_bool * NO_coverage_mask_pad[2:,:-2]
    pairs3_bool=CO_coverage_bool * NO_coverage_mask_pad[:-2,2:]
    
    n_pairs = np.sum(pairs1_bool) + np.sum(pairs2_bool) + np.sum(pairs3_bool)
    
    surface_atoms = np.multiply(*surface.shape[:2])                 
     
    active_sites = n_pairs/surface_atoms
    
    print(str(composition),"fractional active sites:",active_sites,"evaluation time (s):",time.time()-t_start)
    
    
    if return_ads:
        CO_ads = CO_energy_grid[CO_coverage_mask]
        NO_ads = NO_energy_grid[NO_coverage_mask]
        H_ads = H_energy_grid[H_coverage_mask]
        return active_sites, CO_ads, NO_ads, H_ads
    else:
        return active_sites



def count_sites_dynamic(composition,P_CO,P_NO,eU=0,H_UPD=True,return_ads=False):
    #Initiate surface
    surface = initiate_surface(composition,metals)
    
    #Predict energies of every site
    CO_energies,CO_site_ids=predict_energies(surface, "CO", metals)
    NO_energies,NO_site_ids=predict_energies(surface, "NO_fcc", metals)
    H_energies,H_site_ids = predict_energies(surface,"H",metals)
    
    
    
    #Adjust free energy by chemical potential from preassure
    CO_energies += kBT * np.log(1/P_CO)
    NO_energies += kBT * np.log(1/P_NO)
    
    #Turn into grids
    CO_energy_grid = CO_energies.reshape(100,100)
    NO_energy_grid = NO_energies.reshape(100,100)
    H_energy_grid = H_energies.reshape(100,100)
    

    
    CO_data = np.hstack((CO_energies.reshape(-1,1),CO_site_ids))
    NO_data = np.hstack((NO_energies.reshape(-1,1),NO_site_ids))
    H_data = np.hstack((H_energies.reshape(-1,1),H_site_ids))
    
    #Only keep negative energies
    CO_data = CO_data[CO_data[:,0]<0]
    NO_data = NO_data[NO_data[:,0]<0]
    
    
    #Coverage masks
    CO_coverage_mask = np.full((100, 100),None)
    NO_coverage_mask = np.full((100, 100),None)
    H_coverage_mask = np.full((100, 100),None)
    
    
    if H_UPD:
        #fill surface with H where adsorption energy is negative
        for (energy, i, j) in H_data:
            i,j=int(i),int(j)
            if energy<=(-eU):
                H_coverage_mask,CO_coverage_mask,NO_coverage_mask = adsorb_H(i,j,H_coverage_mask, CO_coverage_mask, NO_coverage_mask)
    
    
    
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
                
                
    CO_coverage_mask = np.asanyarray(CO_coverage_mask,dtype=bool)
    NO_coverage_mask = np.asanyarray(NO_coverage_mask,dtype=bool)
    H_coverage_mask = np.asanyarray(H_coverage_mask,dtype=bool)
    
    
    #print("CO coverage:",sum(CO_coverage_mask.flatten()))
    #print("NO coverage:",sum(NO_coverage_mask.flatten()))
    #print("H coverage:",sum(H_coverage_mask.flatten()))
    
    
    #Get site ids of where CO has adsorbed
    CO_ads_sites = np.array(np.where(CO_coverage_mask==True)).T
    
    CO_NO_energy_pair = np.empty((0,2))
    
    
    #Pad grids
    NO_coverage_mask_pad = np.pad(NO_coverage_mask,pad_width=((0,1),(0,1)),mode="wrap")
    NO_energy_grid_pad = np.pad(NO_energy_grid,pad_width=((0,1),(0,1)),mode="wrap")
    
    
    #Get CO-NO pairs of catalytic neighboring sites
    for (i,j) in CO_ads_sites:
        if NO_coverage_mask_pad[i-1,j-1] == True:
            E_CO = CO_energy_grid[i,j]#CO_energies[i*100+j]
            E_NO = NO_energy_grid_pad[i-1,j-1] #NO_energies[(i-1)*100+(j-1)]
            CO_NO_energy_pair = np.vstack((CO_NO_energy_pair,np.array([[E_CO,E_NO]])))
        if NO_coverage_mask_pad[i-1,j+1] == True:
            E_CO = CO_energy_grid[i,j]#CO_energies[i*100+j]
            E_NO = NO_energy_grid_pad[i-1,j+1] #NO_energies[(i-1)*100+(j+1)]
            CO_NO_energy_pair = np.vstack((CO_NO_energy_pair,np.array([[E_CO,E_NO]])))
        if NO_coverage_mask_pad[i+1,j-1] == True:
            E_CO = CO_energy_grid[i,j] #CO_energies[i*100+j]
            E_NO = NO_energy_grid_pad[i+1,j-1] #NO_energies[(i+1)*100+(j-1)]
            CO_NO_energy_pair = np.vstack((CO_NO_energy_pair,np.array([[E_CO,E_NO]])))
            
    
    surface_atoms = np.multiply(*surface.shape[:2])
    n_CO_NO_pairs = len(CO_NO_energy_pair)
    
    if return_ads:
        CO_ads = CO_energy_grid[CO_coverage_mask]
        NO_ads = NO_energy_grid[NO_coverage_mask]
        H_ads = H_energy_grid[H_coverage_mask]
        return n_CO_NO_pairs/surface_atoms, CO_ads, NO_ads, H_ads
    else:
        return n_CO_NO_pairs/surface_atoms
    
    
    
    
    
def MC1(composition,P_CO,P_NO,max_iterations,replacement_factor=1.0,eU=0,H_UPD=True,return_ads=False):
    
    start = time()
    
    #Initiate surface
    surface = initiate_surface(composition,metals)
    
    #Predict energies of every site
    CO_energies,CO_site_ids=predict_energies(surface, "CO", metals)
    NO_energies,NO_site_ids=predict_energies(surface, "NO_fcc", metals)
    H_energies,H_site_ids = predict_energies(surface,"H",metals)
    
    
    #Adjust free energy by chemical potential from preassure
    CO_energies += kBT * np.log(1/P_CO)
    NO_energies += kBT * np.log(1/P_NO)
    
    #Turn into grids
    CO_energy_grid = CO_energies.reshape(100,100)
    NO_energy_grid = NO_energies.reshape(100,100)
    H_energy_grid = H_energies.reshape(100,100)
    
    CO_data = np.hstack((CO_energies.reshape(-1,1),CO_site_ids))
    NO_data = np.hstack((NO_energies.reshape(-1,1),NO_site_ids))
    H_data = np.hstack((H_energies.reshape(-1,1),H_site_ids))
    
    #Only keep negative energies
    CO_data = CO_data[CO_data[:,0]<0]
    NO_data = NO_data[NO_data[:,0]<0]
    
    if len(CO_data)==0 or len(NO_data)==0:
        return 0
    
    #Coverage masks
    CO_coverage_mask = np.full((100, 100),None)
    NO_coverage_mask = np.full((100, 100),None)
    H_coverage_mask = np.full((100, 100),None)
    
    
    if H_UPD:
        #fill surface with H where adsorption energy is negative
        for (energy, i, j) in H_data:
            i,j=int(i),int(j)
            if energy<=(-eU):
                H_coverage_mask,CO_coverage_mask,NO_coverage_mask = adsorb_H(i,j,H_coverage_mask, CO_coverage_mask, NO_coverage_mask)
    
    
    
    P = P_CO + P_NO
    
    p_CO = P_CO/P
    p_NO=P_NO/P

    energy1=0

    for iteration in range(max_iterations):
    #while len(CO_data)>0 or len(NO_data)>0:
        r = np.random.rand()
        if r<= p_CO:
        #if r<=p_CO and len(CO_data)>0:
        #if len(CO_data)>0 and np.min(CO_data[:,0])<=np.min(NO_data[:,0]):
            #p=np.exp(-CO_data[:,0]/kBT - abs(np.min(CO_data[:,0])/kBT))
            #p=p/np.sum(p)
    
            ind = np.random.choice(len(CO_data))#,p=p)
            energy, i, j = CO_data[ind]
            i,j = int(i),int(j)
            if CO_coverage_mask[i,j] and iteration%1000!=0: continue
            #CO_data = np.delete(CO_data, ind,axis=0)
            block_vectors = block_indices(i, j, "top")
            #NO_blocked = np.any([NO_coverage_mask[ib,jb] for (ib,jb) in block_vectors])
            if H_UPD:
                H_blocked = np.any([H_coverage_mask[ib,jb] for (ib,jb) in block_vectors])
                if H_blocked and CO_coverage_mask[i,j] is None: print("error")
                if H_blocked and iteration%1000!=0: continue
            
    
            if CO_coverage_mask[i,j] is None:
                CO_coverage_mask,NO_coverage_mask=adsorb_CO(i, j, CO_coverage_mask, NO_coverage_mask)
    
            #elif CO_coverage_mask[i,j]==False and NO_blocked:
            elif np.random.rand() < replacement_factor:
                
                E=0
                NO_blocking_ind = np.empty((0,2),dtype="int")
                for (i_b,j_b) in block_vectors:
                    if NO_coverage_mask[i_b,j_b]:
                        E += NO_energy_grid[i_b,j_b]
                        NO_blocking_ind = np.vstack((NO_blocking_ind,np.array([[i_b,j_b]])))
                
                Delta_E = CO_energy_grid[i,j] - E
                if Delta_E<0 or np.random.rand() <= np.exp(-Delta_E/kBT):
                    #unblock
                    for i_b,j_b in NO_blocking_ind:
                        i_ub,j_ub = block_indices(i_b, j_b, "fcc").T
                        CO_coverage_mask[i_ub,j_ub]=None
                    CO_coverage_mask,NO_coverage_mask=adsorb_CO(i, j, CO_coverage_mask, NO_coverage_mask)
                    #if Delta_E>0: print(Delta_E,np.exp(-Delta_E/kBT))
        else:        
        #elif len(NO_data)>0:
            #p=np.exp(-NO_data[:,0]/kBT - abs(np.min(NO_data[:,0])/kBT))
    
            #p=p/np.sum(p)
            ind = np.random.choice(len(NO_data))#,p=p)
            energy, i, j = NO_data[ind]
            i,j = int(i),int(j)
            if NO_coverage_mask[i,j] and iteration%1000!=0: continue
            #NO_data = np.delete(NO_data, ind,axis=0)
            block_vectors = block_indices(i, j, "fcc")
            #CO_blocked = np.any([CO_coverage_mask[ib,jb] for (ib,jb) in block_vectors])
            if H_UPD:
                H_blocked = H_coverage_mask[i,j]==True
                if H_blocked and CO_coverage_mask[i,j] is None: print("error")
                if H_blocked and iteration%1000!=0: continue
    
    
            if NO_coverage_mask[i,j] is None:
                CO_coverage_mask,NO_coverage_mask=adsorb_NO(i, j, CO_coverage_mask, NO_coverage_mask)
            
            #elif NO_coverage_mask[i,j]==False and CO_blocked:
            elif np.random.rand() < replacement_factor:
                
                E=0
                CO_blocking_ind = np.empty((0,2),dtype="int")
                for (i_b,j_b) in block_vectors:
                    if CO_coverage_mask[i_b,j_b]:
                        E += CO_energy_grid[i_b,j_b]
                        CO_blocking_ind = np.vstack((CO_blocking_ind,np.array([[i_b,j_b]])))
                        
                Delta_E = NO_energy_grid[i,j] - E
                if Delta_E<0 or np.random.rand() <= np.exp(-Delta_E/kBT):
                    #unblock
                    for i_b,j_b in CO_blocking_ind:
                        i_ub,j_ub = block_indices(i_b, j_b, "top").T
                        NO_coverage_mask[i_ub,j_ub]=None
                    CO_coverage_mask,NO_coverage_mask=adsorb_NO(i, j, CO_coverage_mask, NO_coverage_mask)
                    
                    #if Delta_E>0: print(Delta_E,np.exp(-Delta_E/kBT))
        #if iteration%100000==0:
        #    kBT = kB*300
        #    print(iteration)
        #else:
        #    kBT*=np.exp(-1/10000)
        
        if iteration%1000==0:
            CO_ads = CO_energy_grid[np.asanyarray(CO_coverage_mask,dtype=bool)]
            NO_ads = NO_energy_grid[np.asanyarray(NO_coverage_mask,dtype=bool)]
            H_ads = H_energy_grid[np.asanyarray(H_coverage_mask,dtype=bool)]
            energy2 = np.sum(CO_ads)+np.sum(NO_ads)+np.sum(H_ads)
            
            Delta_energy = abs(energy2-energy1)
            if Delta_energy < 0.001: 
                break
            else: energy1 = energy2
        
    
    CO_coverage_bool = np.asanyarray(CO_coverage_mask,dtype=bool)
    NO_coverage_bool = np.asanyarray(NO_coverage_mask,dtype=bool)
    H_coverage_bool = np.asanyarray(H_coverage_mask,dtype=bool)

    #Pad grids
    NO_coverage_mask_pad = np.pad(NO_coverage_bool,pad_width=((1,1),(1,1)),mode="wrap")
    NO_energy_grid_pad = np.pad(NO_energy_grid,pad_width=((1,1),(1,1)),mode="wrap")


    pairs1_bool=CO_coverage_bool * NO_coverage_mask_pad[:-2,:-2]
    pairs2_bool=CO_coverage_bool * NO_coverage_mask_pad[2:,:-2]
    pairs3_bool=CO_coverage_bool * NO_coverage_mask_pad[:-2,2:]
    
    n_pairs = np.sum(pairs1_bool) + np.sum(pairs2_bool) + np.sum(pairs3_bool)
    
    surface_atoms = np.multiply(*surface.shape[:2])                 
     
    active_sites = n_pairs/surface_atoms
    
    print(str(composition),"fractional active sites:",active_sites,"evaluation time (s):",time.time()-start)
    
    if return_ads:
        CO_ads = CO_energy_grid[CO_coverage_bool]
        NO_ads = NO_energy_grid[NO_coverage_bool]
        H_ads = H_energy_grid[H_coverage_bool]
        return active_sites, CO_ads, NO_ads, H_ads
    else:
        return active_sites





