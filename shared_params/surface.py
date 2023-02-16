import numpy as np
import iteround
from joblib import load
import itertools as it



def initiate_surface(f,metals,size=(100,100),n_layers=3):
    
    #Number of atoms in surface
    n_surface_atoms = np.prod(size)
    #Total number of atoms
    n_atoms = n_surface_atoms*n_layers
    #number of each metal
    n_each_metal = f*n_atoms
    #Round to integer values while maintaining sum
    n_each_metal=iteround.saferound(n_each_metal, 0)
    
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
    reg = load(f'../predict_sites/{adsorbate}/{adsorbate}.joblib')
    
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



def fill_surface(CO_energies,CO_site_ids,NO_energies,NO_site_ids):
    
    #Append index to data
    CO_id = np.ones((len(CO_energies),1))
    NO_id = np.ones((len(NO_energies),1))*2
    CO = np.hstack((CO_energies.reshape(-1,1),CO_site_ids,CO_id))
    NO = np.hstack((NO_energies.reshape(-1,1),CO_site_ids,NO_id))

    #Combine data
    data_array=np.vstack((CO,NO))
    
    #Sort data by lowest energy
    data_array = data_array[data_array[:, 0].argsort()]
    
    #Prepare grids
    fcc_grid = np.zeros((100,100))
    top_grid = fcc_grid.copy()
    
    
    block_fcc_vectors = np.array([[-1,0],[0,-1],[0,0]])
    block_top_vectors = np.array([[-1,0],[-1,1],[1,0]])

    
    for (energy,i,j,idx) in data_array:
        if energy>0: break
        i,j,idx = int(i),int(j),int(idx)
        ij_vec = np.array([i,j])
    
            
        if idx==1:
            if top_grid[i,j]==0:
                top_grid[i,j] = energy
                #CO_energies = np.append(CO_energies,energy)
                
                #Block sites
                ij_block_vectors = ij_vec + block_fcc_vectors
                for (i_b,j_b) in ij_block_vectors:
                    fcc_grid[i_b,j_b] = 1
            else:
                continue
        
        elif idx==2:
            if fcc_grid[i,j]==0:
                fcc_grid[i,j] = energy
                #NO_energies = np.append(NO_energies,energy)
                
                #Block sites
                ij_block_vectors = ij_vec + block_top_vectors
                for (i_b,j_b) in ij_block_vectors:
                    if i_b == 100:
                        i_b=0
                    elif j_b == 100:
                        j_b=0
                    top_grid[i_b,j_b] = 1
    
    return top_grid,fcc_grid



def characterize_sites(fcc_grid,top_grid):
    CO_ids = np.array(np.nonzero(top_grid<0)).T
    CO_NO_energy_pair = np.empty((0,2))
    
    
    #pad grids
    fcc_grid = np.pad(fcc_grid,pad_width=1,mode="wrap")
    top_grid = np.pad(top_grid,pad_width=1,mode="wrap")
    CO_ids+=1
    
    NO_ads_mask = fcc_grid < 0
    
    #Get CO-NO pairs of catalytic neighboring sites
    for (i,j) in CO_ids:
        if fcc_grid[i-1,j-1] < 0:
            E_CO = E_CO = top_grid[i,j]#CO_energies[i*100+j]
            E_NO = fcc_grid[i-1,j-1] #NO_energies[(i-1)*100+(j-1)]
            NO_ads_mask[i-1,j-1] = False
            CO_NO_energy_pair = np.vstack((CO_NO_energy_pair,np.array([[E_CO,E_NO]])))
        if fcc_grid[i-1,j+1] < 0:
            E_CO = E_CO = top_grid[i,j]#CO_energies[i*100+j]
            E_NO = fcc_grid[i-1,j+1] #NO_energies[(i-1)*100+(j+1)]
            NO_ads_mask[i-1,j+1] = False
            CO_NO_energy_pair = np.vstack((CO_NO_energy_pair,np.array([[E_CO,E_NO]])))
        if fcc_grid[i+1,j-1] < 0:
            E_CO = top_grid[i,j] #CO_energies[i*100+j]
            E_NO = fcc_grid[i+1,j-1] #NO_energies[(i+1)*100+(j-1)]
            NO_ads_mask[i+1,j-1] = False
            CO_NO_energy_pair = np.vstack((CO_NO_energy_pair,np.array([[E_CO,E_NO]])))
            
            
        fcc_energies=fcc_grid[1:-1,1:-1].flatten()
        NO_ads_mask = NO_ads_mask[1:-1,1:-1].flatten()
        
        #Remaining fcc sites which are not paired with CO
        NH3_site_energies = fcc_energies[NO_ads_mask]
        
        return CO_NO_energy_pair, NH3_site_energies
    