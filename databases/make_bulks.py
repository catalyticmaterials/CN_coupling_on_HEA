from ase.build import bulk
from ase.db import connect
import numpy as np
from copy import deepcopy

db = connect('bulks.db')

lats_init = dict(Ag = 4.223,
				 Au = 4.229,
				 Cu = 3.697,
				 Pd = 3.9814,
				 Pt = 4.004)


metals = sorted(lats_init.keys())

for metal in metals:

	atoms = bulk(name=metal, crystalstructure='fcc', a=lats_init[metal])

	cell = deepcopy(atoms.cell)

	eps1 = 0.95
	eps2 = 1.05
	
	# Iterate through five unit cell sizes	
	for lat_idx, eps in enumerate(np.linspace(eps1, eps2, 5)):
		atoms.set_cell(cell*eps, scale_atoms=True)
		db.write(atoms, metal=metal, lat_idx=lat_idx)
