from ase.db import connect

"""
with connect("CO_out.db") as db_CO, connect("NO.db") as db_NO:
    for row in db_CO.select():
        atoms = db_CO.get_atoms(row.id)
        symbols = atoms.get_chemical_symbols()
        C_ind=symbols.index("C")
        symbols[C_ind] = "N"
        atoms.set_chemical_symbols(symbols)
        db_NO.write(atoms,**{"slab_idx": row.slab_idx,"site_idx": row.site_idx})
        


with connect("single_element_CO_out.db") as db_CO, connect("single_element_NO.db") as db_NO:
    for row in db_CO.select():
        atoms = db_CO.get_atoms(row.id)
        symbols = atoms.get_chemical_symbols()
        C_ind=symbols.index("C")
        symbols[C_ind] = "N"
        atoms.set_chemical_symbols(symbols)
        db_NO.write(atoms,**{"metal": row.metal})
        
        
        """
"""       
with connect("H.db") as db_H, connect("NO_fcc.db") as db_NO:
    for row in db_H.select():
        atoms = db_H.get_atoms(row.id)
        symbols = atoms.get_chemical_symbols()
        H_ind=symbols.index("H")
        symbols[H_ind] = "N"
        atoms.set_chemical_symbols(symbols)
        atoms.append("O")
        atoms.positions[-1] = atoms.positions[H_ind] + (0,0,1.2)
        
        db_NO.write(atoms,**{"slab_idx": row.slab_idx,"site_idx": row.site_idx})
"""  

with connect("single_element_H.db") as db_H, connect("single_element_NO_fcc.db") as db_NO:
    for row in db_H.select():
        
        atoms = db_H.get_atoms(row.id)
        symbols = atoms.get_chemical_symbols()
        H_ind=symbols.index("H")
        symbols[H_ind] = "N"
        atoms.set_chemical_symbols(symbols)
        atoms.append("O")
        atoms.positions[-1] = atoms.positions[H_ind] + (0,0,1.2)
        
        db_NO.write(atoms,**{"metal": row.metal})