from ase.build import molecule
from ase.db import connect

with connect("molecules.db") as db:
    for mol in ["NO","CO","H"]:

        atoms = molecule(mol)

        atoms.set_cell((20,20,20))

        atoms.set_pbc((1,1,1))

        db.write(atoms,**{"molecule":mol})

