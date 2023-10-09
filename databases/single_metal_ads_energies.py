import numpy as np
from ase.db import connect
import sys
sys.path.append('..')
from scripts import metals


G_corr = {
    'NO': 0.56,
    'CO': 0.40,
    'H': 0.2
}

print('Molecules:')
molecule_energies = []
with connect('molecules_out.db') as mol_db:
    for row in mol_db.select():
        molecule_energies.append(row.energy)
        print(row.molecule,row.energy)

# Apply RPBE coccetion for gas phase NO
molecule_energies[0] += 0.29

print('\nSlabs:')
slab_energies = []
with connect('single_element_slabs_out.db') as slab_db:
    for row in slab_db.select():
        slab_energies.append(row.energy)
        print(row.metal, row.energy)


print('\nNO:')
print('DE, DG')
with connect('single_element_NO_fcc_out.db') as NO_db:
    for row in NO_db.select():
        E  = row.energy - slab_energies[row.id-1] - molecule_energies[0]
        print(f'{row.metal}',E, E + G_corr['NO'])


print('\nCO:')
print('DE, DG')
with connect('single_element_CO_out.db') as CO_db: 
    for row in CO_db.select():
        E = row.energy - slab_energies[row.id-1] - molecule_energies[1]
        print(f'{row.metal}',E, E + G_corr['CO'])



print('\nH:')
print('DE, DG')
with connect('single_element_H_out.db') as H_db:
    for row in H_db.select():
        E = row.energy - slab_energies[row.id-1] - 0.5*molecule_energies[2]
        print(f'{row.metal}',E, E + G_corr['H'])

