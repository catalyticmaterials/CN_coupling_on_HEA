import numpy as np
from kinetic_model import urea_rate, urea_conversion
from surface import predict_energies, initiate_surface

metals = ['Ag','Au', 'Cu', 'Pd','Pt']

def activity(composition,eU,P_NO=1.0):
    
    #Initiate surface
    surface = initiate_surface(composition,metals)
    
    #Predict energies of every site
    CO_energies,CO_site_ids=predict_energies(surface, "CO", metals)
    NO_energies,NO_site_ids=predict_energies(surface, "NO_fcc", metals)
    
    top_energy_grid = CO_energies.reshape(100,100,1)
    fcc_energy_grid = NO_energies.reshape(100,100,1)
    
    fcc_energy_grid_padded = np.pad(fcc_energy_grid,pad_width=((1,1),(1,1),(0,0)),mode="wrap")
    
    
    energy_pairs_1 = np.append(top_energy_grid,fcc_energy_grid_padded[:-2,:-2],axis=2)
    energy_pairs_2 = np.append(top_energy_grid,fcc_energy_grid_padded[:-2,2:],axis=2)
    energy_pairs_3 = np.append(top_energy_grid,fcc_energy_grid_padded[2:,:-2],axis=2)
    
    energy_pairs=np.concatenate((energy_pairs_1.reshape(-1,2),energy_pairs_2.reshape(-1,2),energy_pairs_3.reshape(-1,2)))
    
    #count energy pairs within urea area
    mask_CO = (energy_pairs[:,0] <= 0) * (energy_pairs[:,0] >= -0.7)
    mask_NO = (energy_pairs[:,1] <= 0) * (energy_pairs[:,1] >= -0.6)
    mask = mask_CO * mask_NO
    
    urea_energy_pairs = energy_pairs[mask]
    
    #urea_sites = len(urea_energy_pairs)
    
    #print("Urea site-pairs: ",len(urea_energy_pairs))
    
    surface_atoms = np.multiply(*surface.shape[:2])
    
    urea_rates = np.array([urea_rate(DG_CO, DG_NO, eU,P_NO) for (DG_CO,DG_NO) in urea_energy_pairs])
    
    #rate_pr_atom = np.sum(urea_rates)/surface_atoms
    
    #print("rate per atom: ",rate_pr_atom)
    
    
    urea_conversions = np.array([urea_conversion(DG_CO, DG_NO, eU,P_NO) for (DG_CO,DG_NO) in urea_energy_pairs])
    
    #conversion_pr_site= np.sum(urea_conversions)/urea_sites
    
    #print("conversion per site: ",conversion_pr_site)
    
    
    factored_urea_rate_pr_atom=np.sum(urea_rates*urea_conversions)/surface_atoms
    
    #print("Factored rate per atom",factored_urea_rate_pr_atom)
    
    return factored_urea_rate_pr_atom