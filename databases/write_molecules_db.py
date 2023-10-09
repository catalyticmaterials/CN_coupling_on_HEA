from ase.build import molecule
from ase.db import connect

with connect("molecules.db") as db:
    for mol in ["NO","CO","H2"]:

        atoms = molecule(mol)

        atoms.set_cell((20,20,20))

        atoms.center()

        atoms.set_pbc((1,1,1))

        db.write(atoms,**{"molecule":mol})

    atoms = molecule('NO')

    atoms.set_cell((20,20,20))

    atoms.center()

    atoms.set_pbc((1,1,1))

    atoms.set_initial_magnetic_moments((0,0))

    db.write(atoms,**{"molecule":mol})
