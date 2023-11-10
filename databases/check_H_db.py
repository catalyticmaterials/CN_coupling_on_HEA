import numpy as np
from ase.db import connect
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys
sys.path.append('..')
from scripts import metal_colors,metals
from scripts.site_position_functions import get_nearest_sites_in_xy, get_site_pos_in_xy

H_color = np.array([179,227,245])/255
not_fcc=0
energies = []

# Connect to database with atomic structures of *H
with connect('slabs_out.db') as slab_db,\
    connect('H_out.db') as db,\
    connect('molecules_out.db') as gas_db:

    E_H = 0.5*gas_db._get_row(3).energy
	
	# Iterate through rows in database
    for row in db.select():
		# Get atoms object
        atoms = db.get_atoms(row.id)
		
		# Repeat atoms object
        atoms_3x3 = atoms.repeat((3,3,1))
		
		# Get chemical symbols of atoms
        symbols = np.asarray(atoms_3x3.get_chemical_symbols())
		
		# Get index of central hydrogen atom
        idx_H= np.nonzero(symbols == 'H')[0][4]
		
		# Get position of hydrogen atom
        pos_H = atoms_3x3.positions[idx_H]
		
		# Get indices of the 1st layer atoms
        ids_1st = np.array([atom.index for atom in atoms_3x3 if atom.tag == 1])
		
        # Get posittion of atoms in 1st layer
        pos_1st=atoms_3x3.positions[ids_1st]
        
        #Get position of each site type in xy-plane from the 1st layer atom position
        sites_xy = get_site_pos_in_xy(pos_1st)
        fcc_sites_pos, hcp_sites_pos, bridge_sites_pos, ontop_sites_pos = sites_xy
        #Get site type and min dist site id.
        nearest_site, min_dist_ids = get_nearest_sites_in_xy(*sites_xy,pos_H)
        
        # If nearest site is bridge, locate the next nearest site
        if nearest_site=="bridge": 
            fcc_dist = np.linalg.norm(fcc_sites_pos[min_dist_ids[0]][:2] - pos_H[:2])
            hcp_dist =np.linalg.norm(hcp_sites_pos[min_dist_ids[1]][:2] - pos_H[:2])
            top_dist =np.linalg.norm(ontop_sites_pos[min_dist_ids[3]][:2] - pos_H[:2])
            site_names = ["fcc","hcp","ontop"]
            nearest_site = site_names[np.argmin([fcc_dist,hcp_dist,top_dist])]
        
        # Skip structure if nearest site is not fcc
        if nearest_site!="fcc":
            not_fcc+=1
            # print(nearest_site,row_ads.id)
            continue
        

        energies.append(row.energy - slab_db.get(slab_idx=row.slab_idx).energy - E_H)


# Get adsorption energies on single metal
single_metal_energies = []
with connect('single_element_H_out.db') as db, connect('single_element_slabs_out.db') as slab_db:
    for row_ads, row_slab in zip(db.select(),slab_db.select()):
        single_metal_energies.append(row_ads.energy - row_slab.energy - E_H)


# print number og skipped structures
print('Not fcc',not_fcc)

# Initiate Figure
fig,ax = plt.subplots(figsize=(6,2))

# Plot histogram of adsorption energies
ax.hist(energies,bins=50,histtype='step',color=H_color)
ax.hist(energies,bins=50,alpha=0.4,color=H_color)

# Show single metal dE as lines
ylim = ax.get_ylim()
ax.vlines(single_metal_energies,ylim[0],ylim[1],ls=':',colors=[metal_colors[metal] for metal in metals])
for metal, energy in zip(metals,single_metal_energies):
    if metal=='Au':
        ax.text(energy,ylim[1],metal,color=metal_colors[metal],va='bottom',ha='left')
    elif metal=='Ag':
        ax.text(energy,ylim[1],metal,color=metal_colors[metal],va='bottom',ha='right')
    else:
        ax.text(energy,ylim[1],metal,color=metal_colors[metal],va='bottom',ha='center')

# Hide the left, right and top spines
ax.spines[['left','right', 'top']].set_visible(False)
ax.tick_params(left=False)
ax.set_yticks([])
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MultipleLocator(0.5))

ax.set_xlabel('$\Delta E_{*H}$ (eV)')

plt.tight_layout()
plt.savefig('H_ads.png',dpi=600)