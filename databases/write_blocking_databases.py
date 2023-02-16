from ase.db import connect
"""
n=0
with connect("NO_fcc.db") as db_NO, connect("NO_fcc_block.db") as db_B:
    for row in db_NO.select():
        if row.site_idx!=4:
            continue
        atoms = db_NO.get_atoms(row.id)
        atoms2 = db_NO.get_atoms(row.id+1)
        symbols = atoms.get_chemical_symbols()
        symbols2 = atoms2.get_chemical_symbols()
        N_ind=symbols2.index("N")
        O_ind=symbols2.index("O")
        
        N_pos=atoms2.get_positions()[-2]
        O_pos=atoms2.get_positions()[-1]
        
        #symbols.append(["N","O"])

        #atoms.set_chemical_symbols(symbols)
        atoms.append("N")
        atoms.append("O")
        atoms.positions[-2] = N_pos
        atoms.positions[-1] = O_pos
        db_B.write(atoms,**{"slab_idx": row.slab_idx,"site_idx": row.site_idx})
        n+=1
        if n==10: break
    
"""

with connect("NO_fcc.db") as db_NO:
    atoms = db_NO.get_atoms(2)

    symbols = atoms.get_chemical_symbols()
    N_ind=symbols.index("N")
    O_ind=symbols.index("O")
    
    N_pos=atoms.get_positions()[-2]
    O_pos=atoms.get_positions()[-1]



with connect("single_element_NO_fcc.db") as db_NO, connect("single_element_NO_fcc_block.db") as db_B:
    for row in db_NO.select():
        atoms = db_NO.get_atoms(row.id)

        
        #symbols.append(["N","O"])

        #atoms.set_chemical_symbols(symbols)
        atoms.append("N")
        atoms.append("O")
        atoms.positions[-2] = N_pos
        atoms.positions[-1] = O_pos
        db_B.write(atoms,**{"metal": row.metal})
