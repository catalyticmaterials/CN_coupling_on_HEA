import numpy as np
from ase.db import connect
from ase.data import covalent_radii,vdw_radii

not_ontop=0


with connect("CO_out.db") as db:
    for row in db.select():
        atoms=db.get_atoms(row.id)
        site_idx=row.site_idx
        
        # Repeat atoms object
        atoms_3x3 = atoms.repeat((3,3,1))
		
		# Get chemical symbols of atoms
        symbols = np.asarray(atoms_3x3.get_chemical_symbols())
		
		# Get index of central hydrogen atom
        idx_C = np.nonzero(symbols == 'C')[0][4]
		
		# Get position of carbon atom
        pos_C = atoms_3x3.positions[idx_C]
		
		# Get indices of the 1st layer atoms
        ids_1st = np.array([atom.index for atom in atoms_3x3 if atom.tag == 1])
		
		# Get squared distances to the 1st layer atoms from the hydrogen atom
        dists_sq = np.sum((atoms_3x3.positions[ids_1st] - pos_C)**2, axis=1)
		
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
        r_C = covalent_radii[6]
        
        #Set reference distance
        r_ref = r + r_C
        
        #Get number of distances below reference +20%
        n = np.sum(dist_3 < r_ref*1.2)

        if n>1:
            not_ontop+=1
            print(row.id)

#print(ontop,bridge,hollow)
#print(sum([ontop,bridge,hollow]))
print("not ontop:",not_ontop)