from ase.db import connect
import numpy as np


metals = ['Ag','Au', 'Cu', 'Pd','Pt']

with connect("single_element_slabs_out.db") as slab_db, connect("single_element_NO_fcc_out.db") as NO_db, connect("single_element_NO_fcc_block_out.db") as block_db:
    for metal in metals:
        E_slab = slab_db.get(metal=metal).energy
        E_NO = NO_db.get(metal=metal).energy
        E_block = block_db.get(metal=metal).energy
        
        print(metal,(E_block-E_NO)-(E_NO-E_slab))
