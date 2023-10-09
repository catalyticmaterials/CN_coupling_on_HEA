import numpy as np
from ase.build import fcc111, add_adsorbate,molecule
from ase.db import connect
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, covalent_radii


metals = ["Ag", "Au", "Cu", "Pd", "Pt"]


lattice_parameters = {
  "Ag": 4.2461,
  "Au": 4.2502,
  "Cu": 3.7031,
  "Pd": 4.0122,
  "Pt": 4.0203
}



# Specify size of slab
size = (3,3,5)
n_atoms = np.prod(size)
n_atoms_surface = np.prod(size[:2])

# Connect to database
with connect('single_element_slabs2.db') as db_slab, connect('single_element_CO_2.db') as db_CO:

    for metal in metals:
	
        # Get average lattice parameter of surface atoms
        lat = lattice_parameters[metal]
        
        # Make slab
        slab = fcc111(metal, size=size, a=lat, vacuum=10.)

        # Fix all but the two top layers of atoms
        constraint = FixAtoms(indices=[atom.index for atom in slab if atom.tag > 2])
        slab.set_constraint(constraint)

        # Save slab atoms object to database
        db_slab.write(slab, metal=metal)

        h = covalent_radii[atomic_numbers[metal]] + covalent_radii[6] - 0.15

        add_adsorbate(slab,molecule('CO'),height=h,mol_index=1)
        
        db_CO.write(slab,metal=metal)