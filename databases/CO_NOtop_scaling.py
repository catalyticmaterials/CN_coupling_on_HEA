import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from ase.db import connect
from ase.data import covalent_radii
from scipy.stats import linregress
import sys
sys.path.append('..')
from scripts import metal_colors,metals

CO_energies = []
NO_energies = []

site_symbols = []

with connect('CO_out.db') as CO_db, connect('NO_out_sorted.db') as NO_db, connect('slabs_out.db') as slab_db:
    for row_CO, row_NO in zip(CO_db.select(),NO_db.select()):
        
        # CO
        atoms=CO_db.get_atoms(row_CO.id)
        site_idx=row_CO.site_idx
        
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
            continue

        # Get the closest surface atom to the carbon (the on-top adsorption site)
        idx_site = ids_1st[np.argmin(dists_sq)]		

        # Get symbol of adsorption site
        symbol_site = symbols[idx_site]



        # NO
        atoms=NO_db.get_atoms(row_NO.id)
        site_idx=row_NO.site_idx
        
        # Repeat atoms object
        atoms_3x3 = atoms.repeat((3,3,1))
		
		# Get chemical symbols of atoms
        symbols = np.asarray(atoms_3x3.get_chemical_symbols())
		
		# Get index of central hydrogen atom
        idx_N = np.nonzero(symbols == 'N')[0][4]
		
		# Get position of carbon atom
        pos_N = atoms_3x3.positions[idx_N]
		
		# Get indices of the 1st layer atoms
        ids_1st = np.array([atom.index for atom in atoms_3x3 if atom.tag == 1])
		
		# Get squared distances to the 1st layer atoms from the hydrogen atom
        dists_sq = np.sum((atoms_3x3.positions[ids_1st] - pos_N)**2, axis=1)
		
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
        r_C = covalent_radii[7]
        
        #Set reference distance
        r_ref = r + r_C
        
        #Get number of distances below reference +20%
        n = np.sum(dist_3 < r_ref*1.2)

        if n>1:
            continue

        
        # Get the closest surface atom to the carbon (the on-top adsorption site)
        idx_site = ids_1st[np.argmin(dists_sq)]	

        # Check it is the same site 
        if symbol_site==symbols[idx_site]:
            assert(row_CO.slab_idx==row_NO.slab_idx)

            slab_energy = slab_db.get(slab_idx=row_CO.slab_idx).energy

            CO_energies.append(row_CO.energy - slab_energy)

            NO_energies.append(row_NO.energy - slab_energy)

            site_symbols.append(symbol_site)


colors = [metal_colors[metal] for metal in site_symbols]

fig,ax = plt.subplots(figsize=(4,4))

ax.scatter(CO_energies,NO_energies,c=colors)

handles = []
for metal in metals:
    handles.append(Line2D([0], [0], marker='o', color="w", label=metal,markerfacecolor=metal_colors[metal], markersize=10))    




ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

a,b,*ignore = linregress(CO_energies,NO_energies)

xlim = ax.get_xlim()
ylim = ax.get_ylim()

line = ax.plot(xlim,[a*xlim[0]+b,a*xlim[1]+b],c='k',label='Scaling')

handles.append(line[0])

ax.legend(handles=handles,loc=2)

ax.set_xlabel('$\Delta E_{*CO}$ (eV)')
ax.set_ylabel('$\Delta E_{*NO}$ (eV)')

ax.text(0.3,0.1,f'$\Delta E_{{*NO}} = {a:.2f}\Delta E_{{*NO}} + {b:.2f}$ eV',transform = ax.transAxes)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)

plt.subplots_adjust(bottom=0.2,left=0.2)
plt.show()
