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

def count_sites(composition,P_CO,P_NO, metals, method, eU=0, n=100, return_ads_energies=False,print_result=False):
    t_start = time()

    # Simulate surface
    (CO_coverage_bool, NO_coverage_bool, H_coverage_bool), (CO_energy_grid, NO_energy_grid, H_energy_grid) = fill_surface(composition,P_CO,P_NO, metals, method, eU=eU, n=n)

    # count sites
    active_sites = count_CO_NO_pairs(CO_coverage_bool, NO_coverage_bool)

    if print_result:
        print(str(np.around(composition,decimals=4)),"fractional active sites:",active_sites,"evaluation time (s):",time()-t_start)

    if return_ads_energies:
        CO_ads = CO_energy_grid[CO_coverage_bool]
        NO_ads = NO_energy_grid[NO_coverage_bool]
        H_ads = H_energy_grid[H_coverage_bool]
        return active_sites, CO_ads, NO_ads, H_ads
    else:
        return active_sites

def get_sites(composition,P_CO,P_NO, metals, method, eU=0, n=100, return_ads_energies=False):

    # Simulate surface
    (CO_coverage_bool, NO_coverage_bool, H_coverage_bool), (CO_energy_grid, NO_energy_grid, H_energy_grid) = fill_surface(composition,P_CO,P_NO, metals, method, eU=eU, n=n)

    # count sites
    CO_NO_energy_pairs = CO_NO_pairs_energy(CO_coverage_bool, NO_coverage_bool,CO_energy_grid, NO_energy_grid)

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

    
    # Number of atoms in the surface layer
    surface_atoms = np.multiply(*CO_coverage_bool.shape)                 

    # Locate pairs
    CO_coverage_mask_pad = np.pad(CO_coverage_bool,pad_width=((1,1),(1,1)),mode="wrap")
    pairs1_bool=NO_coverage_bool * CO_coverage_mask_pad[:-2,:-2]
    pairs2_bool=NO_coverage_bool * CO_coverage_mask_pad[2:,:-2]
    pairs3_bool=NO_coverage_bool * CO_coverage_mask_pad[:-2,2:]

    # Get coupled NO
    N_CN_NO = np.sum(pairs1_bool+pairs2_bool+pairs3_bool)

    # Get number of pairs
    N_CO_NO_pairs = (np.sum(pairs1_bool) + np.sum(pairs2_bool) + np.sum(pairs3_bool))
    
    # unpaired NO
    N_NH3_NO = np.sum(NO_coverage_bool) - N_CN_NO

    return N_H_ads/surface_atoms, N_NH3_NO/surface_atoms, N_CN_NO/surface_atoms, N_CO_NO_pairs/surface_atoms