import numpy as np
from ase.build import fcc111, add_adsorbate,molecule
from ase.db import connect
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, covalent_radii


metals = ['Rh','Ir','Ni','Pd','Rh']
ads = ['CO','CO','CO','CO','NO']
ads_site = ['top','top','fcc','fcc','fcc']


lattice_parameters = {
  "Ag": 4.2113,
  "Al": 4.0674,
  "Au": 4.2149,
  "B": 2.8822,
  "Be": 3.2022,
  "Bi": 5.0699,
  "Cd": 4.5795,
  "Co": 3.5625,
  "Cr": 3.6466,
  "Cu": 3.6901,
  "Fe": 3.6951,
  "Ga": 4.2817,
  "Ge": 4.304,
  "Hf": 4.5321,
  "In": 4.8536,
  "Ir": 3.8841,
  "Mg": 4.5673,
  "Mn": 3.5371,
  "Mo": 4.0204,
  "Nb": 4.2378,
  "Ni": 3.565,
  "Os": 3.8645,
  "Pb": 5.0942,
  "Pd": 3.9814,
  "Pt": 3.9936,
  "Re": 3.9293,
  "Rh": 3.8648,
  "Ru": 3.8285,
  "Sc": 4.6809,
  "Si": 3.8935,
  "Sn": 4.8612,
  "Ta": 4.2504,
  "Ti": 4.158,
  "Tl": 5.0884,
  "V": 3.8573,
  "W": 4.0543,
  "Y": 5.1289,
  "Zn": 3.9871,
  "Zr": 4.562
}



# Specify size of slab
size = (3,3,5)
n_atoms = np.prod(size)
n_atoms_surface = np.prod(size[:2])



# # Connect to database
# with connect('single_element_slabs_extra.db') as db_slab, connect('single_element_ads_extra.db') as db_ads:

#     for i,metal in enumerate(metals):
	
#         # Get average lattice parameter of surface atoms
#         lat = lattice_parameters[metal]
        
#         # Make slab
#         slab = fcc111(metal, size=size, a=lat, vacuum=10.)

#         # Fix all but the two top layers of atoms
#         constraint = FixAtoms(indices=[atom.index for atom in slab if atom.tag > 2])
#         slab.set_constraint(constraint)

#         if i<3:
#             # Save slab atoms object to database
#             db_slab.write(slab, metal=metal)


#         if ads_site[i]=='top':

#             h = covalent_radii[atomic_numbers[metal]] + covalent_radii[atomic_numbers[ads[i][0]]] - 0.15

#             add_adsorbate(slab,molecule(ads[i]),height=h,mol_index=1,position='ontop')
        
#         elif ads_site[i]=='fcc':
#             if ads[i]=='CO':
#                 h = covalent_radii[atomic_numbers[metal]]
#             else:
#                 h = covalent_radii[atomic_numbers[metal]] + 1

#             add_adsorbate(slab,molecule(ads[i]),height=h,mol_index=1,position='fcc')
        
#         db_ads.write(slab,metal=metal)


# Get average lattice parameter of surface atoms
lat = lattice_parameters['Cu']

# Make slab
slab = fcc111('Cu', size=size, a=lat, vacuum=10.)
with connect('CO_on_Cu_unconstrained.db') as db:
    db.write(slab)

    h = covalent_radii[atomic_numbers['Cu']] + covalent_radii[atomic_numbers['C']] - 0.15
    
    add_adsorbate(slab,molecule('CO'),height=h,mol_index=1,position='ontop')

    db.write(slab)