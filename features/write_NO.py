import numpy as np
from ase.db import connect
from collections import Counter
from ase.data import covalent_radii

# Define metals in the considered alloys
metals = ['Ag','Au', 'Cu', 'Pd','Pt']

#Free energy reference to Cu
DG_Cu = +0.45

# Get Cu(111) reference energy
path = '../databases'
with connect(f'{path}/single_element_slabs_out.db') as db_slab,\
	 connect(f'{path}/single_element_NO_angled_out.db') as db_ads:
	 
	E_ref = db_ads.get(metal='Cu').energy - db_slab.get(metal='Cu').energy + DG_Cu

# Set filename
filename = 'NO.csv'

# Write header to file
with open(filename, 'w') as file_:
	file_.write('site, 1st layer, 2nd layer, 3rd layer, adsorption energy (eV), row id ads, row id slab')

# Connect to database with atomic structures of *CO
with connect(f'{path}/slabs_out.db') as db_slab,\
	 connect(f'{path}/NO_out_sorted.db') as db_ads,\
	 open(filename, 'a') as file_:
	
	# Iterate through rows in database
	for row_ads in db_ads.select():
		
		# Get atoms object
		atoms = db_ads.get_atoms(row_ads.id)
		
		# Repeat atoms object
		atoms_3x3 = atoms.repeat((3,3,1))
		
		# Get chemical symbols of atoms
		symbols = np.asarray(atoms_3x3.get_chemical_symbols())
		
		# Get index of central Nitrogen atom
		idx_N = np.nonzero(symbols == 'N')[0][4]
		
		# Get position of Nitrogen atom
		pos_N = atoms_3x3.positions[idx_N]
		
		# Get indices of the 1st layer atoms
		ids_1st = np.array([atom.index for atom in atoms_3x3 if atom.tag == 1])
		
		# Get squared distances to the 1st layer atoms from the carbon atom
		dists_sq = np.sum((atoms_3x3.positions[ids_1st] - pos_N)**2, axis=1)
		
        ## Check if N not on ontop site
        # Get the three shortest distances
		dist_3 = np.sort(np.sqrt(dists_sq))[:3]  
        
        # Get one atom from 4th and 5th layer 
		id_4th = np.array([atom.index for atom in atoms_3x3 if atom.tag == 4])[0]
		id_5th = np.array([atom.index for atom in atoms_3x3 if atom.tag == 5])[7]
        
        # Get distance in z
		z_dist = atoms_3x3.positions[id_4th][2] - atoms_3x3.positions[id_5th][2]
        
        # Set average radius of atoms
		r = z_dist/2
        # Get radius of N
		r_N = covalent_radii[7]
        
        #Set reference distance
		r_ref = r + r_N
        
        #Get number of distances below reference +20%
		n = np.sum(dist_3 < r_ref*1.2)

		if n>1:
			print("Skips row",row_ads.id,". Not ontop site")
			#Continue with next structure if the site is not on-top.
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
		file_.write(f'\n{features_str},{energy:.6f},{row_ads.id},{row_slab.id}')

print(f'[SAVED] {filename}')
