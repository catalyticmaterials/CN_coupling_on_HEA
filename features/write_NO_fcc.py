import numpy as np
from ase.db import connect
from collections import Counter
import itertools as it
from site_position_functions import get_nearest_sites_in_xy, get_site_pos_in_xy

# Define metals in the considered alloys
metals = ['Ag','Au', 'Cu', 'Pd','Pt']

# Get number of metals
n_metals = len(metals)

# Get three-fold hollow site ensembles
ensembles = list(it.combinations_with_replacement(metals, 3))

# Get number of adsorption site ensembles
n_ensembles = len(ensembles)

# Specify the number of zones
n_zones = 2

#Free energy reference to Cu
DG_Cu = +0.45
# Get Cu(111) reference energy
path = '../databases'
with connect(f'{path}/single_element_slabs_out.db') as db_slab,\
	 connect(f'{path}/single_element_NO_fcc_out.db') as db_ads:
	 
	E_ref = db_ads.get(metal='Cu').energy - db_slab.get(metal='Cu').energy + DG_Cu

# Set filename
filename = 'NO_fcc.csv'

# Write header to file
with open(filename, 'w') as file_:
	file_.write('site, 1st layer, 2nd layer, adsorption energy (eV), row id ads, row id slab,site id')

skipped_rows=0
# Connect to database with atomic structures of *CO
with connect(f'{path}/slabs_out.db') as db_slab,\
    connect(f'{path}/NO_fcc_out.db') as db_ads,\
    open(filename, 'a') as file_:
	
	# Iterate through rows in database
    for row_ads in db_ads.select():
		# Check for empty atoms object
        #if row_ads.formula =="":
        #    continue
        #if row_ads.id!=3: continue
		# Get atoms object
        atoms = db_ads.get_atoms(row_ads.id)
		
		# Repeat atoms object
        atoms_3x3 = atoms.repeat((3,3,1))
		
		# Get chemical symbols of atoms
        symbols = np.asarray(atoms_3x3.get_chemical_symbols())
		
		# Get index of central Nitrogen atom
        idx_N= np.nonzero(symbols == 'N')[0][4]
		
		# Get position of hydrogen atom
        pos_N = atoms_3x3.positions[idx_N]
		
		# Get indices of the 1st layer atoms
        ids_1st = np.array([atom.index for atom in atoms_3x3 if atom.tag == 1])
		
        # Get posittion of atoms in 1st layer
        pos_1st=atoms_3x3.positions[ids_1st]
        
		# Get squared distances to the 1st layer atoms from the hydrogen atom
        dists_sq = np.sum((pos_1st - pos_N)**2, axis=1)
		
		# Get the closest 3 surface atoms to the hydrogen
		# (the three-fold hollow adsorption site)
        ids_site = ids_1st[np.argpartition(dists_sq, kth=2)[:3]]
		
		# Get the position of the adsorption site as the mean of the positions
		# of the atoms in the site
        pos_site = np.mean(atoms_3x3.positions[ids_site], axis=0)
		
		# Get indices of the 2nd layer atoms
        ids_2nd = np.array([atom.index for atom in atoms_3x3 if atom.tag == 2])
		
		# Get squared distances from adsorption site to the atoms in each layer
        dists_sq_1st = np.sum((atoms_3x3.positions[ids_1st] - pos_site)**2, axis=1)
        dists_sq_2nd = np.sum((atoms_3x3.positions[ids_2nd] - pos_site)**2, axis=1)
		
        # Get one atom from 4th and 5th layer 
        id_4th = np.array([atom.index for atom in atoms_3x3 if atom.tag == 4])[0]
        id_5th = np.array([atom.index for atom in atoms_3x3 if atom.tag == 5])[0]
        
        # Get distance in z
        z_dist = atoms_3x3.positions[id_4th][2] - atoms_3x3.positions[id_5th][2]
        
        min_dist_2nd = np.sqrt(np.min(dists_sq_2nd))
		
        #print(np.sort(np.sqrt(dists_sq_2nd))[:3],z_dist*1.1)
        #break
        
        #Get position of each site type in xy-plane from the 1st layer atom position
        sites_xy = get_site_pos_in_xy(pos_1st)
        fcc_sites_pos, hcp_sites_pos, bridge_sites_pos, ontop_sites_pos = sites_xy
        #Get site type and min dist site id.
        nearest_site, min_dist_ids = get_nearest_sites_in_xy(*sites_xy,pos_N)
        
        # If nearest site is bridge, locate the next nearest site
        if nearest_site=="bridge": 
            fcc_dist = np.linalg.norm(fcc_sites_pos[min_dist_ids[0]][:2] - pos_N[:2])
            hcp_dist =np.linalg.norm(hcp_sites_pos[min_dist_ids[1]][:2] - pos_N[:2])
            top_dist =np.linalg.norm(ontop_sites_pos[min_dist_ids[3]][:2] - pos_N[:2])
            site_names = ["fcc","hcp","ontop"]
            nearest_site = site_names[np.argmin([fcc_dist,hcp_dist,top_dist])]
            
        if nearest_site!="fcc":
            skipped_rows+=1
            print(nearest_site,row_ads.id)
            continue
        
        """
        if nearest_site == "bridge" or nearest_site=="ontop": 
            fcc_dist = np.linalg.norm(fcc_sites_pos[min_dist_ids[0]][:2] - pos_N[:2])
            hcp_dist =np.linalg.norm(hcp_sites_pos[min_dist_ids[1]][:2] - pos_N[:2])
            site_names = ["fcc","hcp"]
            nearest_site = site_names[np.argmin([fcc_dist,hcp_dist])]
        
        if nearest_site=="hcp":
            skipped_rows+=1
            continue
        """
        pos_site = fcc_sites_pos[min_dist_ids[0]]
        
        
        """
        #Check for hcp vs fcc
        if min_dist_2nd < z_dist*1.1:
            #Get position of each site type in xy-plane from the 1st layer atom position
            sites_xy = get_site_pos_in_xy(pos_1st)
            
            #Get site type and min dist site id.
            nearest_site, min_dist_ids = get_nearest_sites_in_xy(*sites_xy,pos_N)
            
            if nearest_site=="fcc":
                #Set site position to nearest fcc site
                fcc_sites = sites_xy[0]
                pos_site = fcc_sites[min_dist_ids[0]]
            if nearest_site=="hcp":
                #Skip row in database if hcp
                continue
            else:
                #If bridge or ontop then use the fcc site it came from before relaxation
                with connect("../databases/NO_fcc.db") as NO_db:
                    #Get atoms object
                    atoms = NO_db.get_atoms(row_ads.id)
                    
                    # Repeat atoms object
                    atoms_3x3 = atoms.repeat((3,3,1))
            		
            		# Get chemical symbols of atoms
                    symbols = np.asarray(atoms_3x3.get_chemical_symbols())
            		
            		# Get index of central hydrogen atom
                    idx_N = np.nonzero(symbols == 'N')[0][4]
            		
            		# Get position of hydrogen atom
                    pos_N = atoms_3x3.positions[idx_N]
                    
                    # Get indices of the 1st layer atoms
                    ids_1st = np.array([atom.index for atom in atoms_3x3 if atom.tag == 1])
            		
                    # Get posittion of atoms in 1st layer
                    pos_1st=atoms_3x3.positions[ids_1st]
                    
            		# Get squared distances to the 1st layer atoms from the hydrogen atom
                    dists_sq = np.sum((pos_1st - pos_N)**2, axis=1)
            		
            		# Get the closest 3 surface atoms to the hydrogen
            		# (the three-fold hollow adsorption site)
                    ids_site = ids_1st[np.argpartition(dists_sq, kth=2)[:3]]
            		
            		# Get the position of the adsorption site as the mean of the positions
            		# of the atoms in the site
                    pos_site = np.mean(atoms_3x3.positions[ids_site], axis=0)
                    
                    """
                    
		# Sort atom indices in the top 2 layers according to their distance to the adsorption site
        ids_1st_sorted = ids_1st[np.argsort(dists_sq_1st)]
        ids_2nd_sorted = ids_2nd[np.argsort(dists_sq_2nd)]
		
		# Get symbols of adsorption site
        symbols_site = symbols[ids_site]

		# Get the next 3 closest surface atoms
        symbols_1st = symbols[ids_1st_sorted[3:6]]
		
		# Get the 3 closest atoms in the 2nd layer
        symbols_2nd = symbols[ids_2nd_sorted[:3]]
		
		# Get the adsorption ensemble
        ensemble = tuple(sorted(symbols_site))
		
		# Get index of the current ensemble
        idx_ensemble = ensembles.index(ensemble)
		
		# Get the count of each atom in each zone
        count_1st = Counter(symbols_1st)
        count_2nd = Counter(symbols_2nd)
		
		# Initiate list of features
        features = np.zeros(n_ensembles + n_metals * n_zones, dtype=int)
		
		# One-hot encode the adsorption ensemble
        features[idx_ensemble] = 1
		
		# Iterate through element counts in each zone
        for idx_zone, count in enumerate([count_1st, count_2nd]):
			
			# Make element counts into features
            features[n_ensembles + idx_zone * n_metals : n_ensembles + (idx_zone + 1) * n_metals] = [count[m] for m in metals]
		
		# Make features into string	
        features_str = ','.join(map(str, features))
		
		# Get corresponding slab without adsorbate
        row_slab = db_slab.get(slab_idx=row_ads.slab_idx)
		
		# Get adsorption energy as the difference between the slab with and without adsorbate
        energy = row_ads.energy - row_slab.energy - E_ref
		
		# Write features and energy to file
        file_.write(f'\n{features_str},{energy:.6f},{row_ads.id},{row_slab.id},{ids_site%47 - 36}')

print(f'[SAVED] {filename}')
print(skipped_rows)