import numpy as np
from ase.db import connect
from collections import Counter

import sys
sys.path.append('..')
from scripts.site_position_functions import get_nearest_sites_in_xy, get_site_pos_in_xy

# Define metals in the considered alloys
metals = ['Ag','Au', 'Cu', 'Pd','Pt']

#Database path
path = '../databases'

# Set filename
filename = 'CO.csv'

# Write header to file
with open(filename, 'w') as file_:
	file_.write('site, 1st layer, 2nd layer, 3rd layer, adsorption energy (eV), row id ads, row id slab, site id')


#Free energy reference to Cu
DG_Cu = 0.12

# Write single metal features
with connect(f'{path}/single_element_slabs_out.db') as db_slab,\
	 connect(f'{path}/single_element_CO_out.db') as db_ads,\
	open(filename, 'a') as file_:
	
	# Get Cu(111) reference energy
	E_ref = db_ads.get(metal='Cu').energy - db_slab.get(metal='Cu').energy + DG_Cu

	# Iterate through rows in db
	for row_ads in db_ads.select():
		# Get metal index
		metal_index = metals.index(row_ads.metal)

		# make site feature by one-hot encoding metal index
		site_feature = np.zeros(5,dtype=int)
		site_feature[metal_index] = 1

		# Make feature
		features = np.concatenate((site_feature,site_feature*6, site_feature*3, site_feature*3))
		features_str = ','.join(map(str, features))

		# Get corresponding slab without adsorbate
		row_slab = db_slab.get(metal=row_ads.metal)
		
		# Get adsorption energy as the difference between the slab with and without adsorbate
		energy = row_ads.energy - row_slab.energy - E_ref
		
		# Write features and energy to file
		file_.write(f'\n{features_str},{energy:.6f},{row_ads.id},{row_slab.id},0')


# Connect to database with atomic structures of *CO
with connect(f'{path}/slabs_out.db') as db_slab,\
	 connect(f'{path}/CO_out.db') as db_ads,\
	 open(filename, 'a') as file_:
	
	# Iterate through rows in database
	for row_ads in db_ads.select():
		
		# Get atoms object
		atoms = db_ads.get_atoms(row_ads.id)
		
		# Repeat atoms object
		atoms_3x3 = atoms.repeat((3,3,1))
		
		# Get chemical symbols of atoms
		symbols = np.asarray(atoms_3x3.get_chemical_symbols())
		
		# Get index of central carbon atom
		idx_C = np.nonzero(symbols == 'C')[0][4]
		
		# Get position of carbon atom
		pos_C = atoms_3x3.positions[idx_C]
		
		# Get indices of the 1st layer atoms
		ids_1st = np.array([atom.index for atom in atoms_3x3 if atom.tag == 1])
		
		# Get squared distances to the 1st layer atoms from the carbon atom
		dists_sq = np.sum((atoms_3x3.positions[ids_1st] - pos_C)**2, axis=1)

		# Get posittion of atoms in 1st layer
		pos_1st=atoms_3x3.positions[ids_1st]
		# Get position of each site type in xy-plane from the 1st layer atom position
		sites_xy = get_site_pos_in_xy(pos_1st)
		fcc_sites_pos, hcp_sites_pos, bridge_sites_pos, ontop_sites_pos = sites_xy
		# Get site type and min dist site id.
		nearest_site, min_dist_ids = get_nearest_sites_in_xy(*sites_xy,pos_C)

		# If nearest site is bridge, locate the next nearest site
		if nearest_site=="bridge": 
			fcc_dist = np.linalg.norm(fcc_sites_pos[min_dist_ids[0]][:2] - pos_C[:2])
			hcp_dist =np.linalg.norm(hcp_sites_pos[min_dist_ids[1]][:2] - pos_C[:2])
			top_dist =np.linalg.norm(ontop_sites_pos[min_dist_ids[3]][:2] - pos_C[:2])
			site_names = ["fcc","hcp","ontop"]
			nearest_site = site_names[np.argmin([fcc_dist,hcp_dist,top_dist])]
            
		if nearest_site!="ontop":
			print("Skips row",row_ads.id,". Not ontop site")
			# Continue with next structure if the site is not on-top.
			continue
		
		# Get the closest surface atom to the carbon (the on-top adsorption site)
		idx_site = ids_1st[np.argmin(dists_sq)]		
		
		# Get the position of the adsorption site
		pos_site = atoms_3x3.positions[idx_site]
		
		# Get indices of the 2nd layer atoms
		ids_2nd = np.array([atom.index for atom in atoms_3x3 if atom.tag == 2])
		
		# Get indices of the 3rd layer atoms
		ids_3rd = np.array([atom.index for atom in atoms_3x3 if atom.tag == 3])
		
		# Get squared distances from adsorption site to the atoms in each layer
		dists_sq_1st = np.sum((atoms_3x3.positions[ids_1st] - pos_site)**2, axis=1)
		dists_sq_2nd = np.sum((atoms_3x3.positions[ids_2nd] - pos_site)**2, axis=1)
		dists_sq_3rd = np.sum((atoms_3x3.positions[ids_3rd] - pos_site)**2, axis=1)
		
		# Sort atom indices in the top 3 layers according to their distance to the adsorption site
		ids_1st_sorted = ids_1st[np.argsort(dists_sq_1st)]
		ids_2nd_sorted = ids_2nd[np.argsort(dists_sq_2nd)]
		ids_3rd_sorted = ids_3rd[np.argsort(dists_sq_3rd)]
		
		# Get symbol of adsorption site
		symbol_site = symbols[idx_site]
		
		# Get the next 6 closest surface atoms
		symbols_1st = symbols[ids_1st_sorted[1:7]]
		
		# Get the 3 closest atoms in the 2nd layer
		symbols_2nd = symbols[ids_2nd_sorted[:3]]
		
		# Get the 3 to 6 closest atoms in the 3rd layer
		symbols_3rd = symbols[ids_3rd_sorted[3:6]]
		
		# Get the count of each atom in each zone
		count_site = Counter([symbol_site])
		count_1st = Counter(symbols_1st)
		count_2nd = Counter(symbols_2nd)
		count_3rd = Counter(symbols_3rd)
		
		# Make count into a list of features
		features = np.array([[count[m] for m in metals] for count in [count_site, count_1st, count_2nd, count_3rd]]).astype(int).ravel()
		features_str = ','.join(map(str, features))
		
		# Get corresponding slab without adsorbate
		row_slab = db_slab.get(slab_idx=row_ads.slab_idx)
		
		# Get adsorption energy as the difference between the slab with and without adsorbate
		energy = row_ads.energy - row_slab.energy - E_ref
		
		# Write features and energy to file
		file_.write(f'\n{features_str},{energy:.6f},{row_ads.id},{row_slab.id},{idx_site-47*4-36}')

print(f'[SAVED] {filename}')
