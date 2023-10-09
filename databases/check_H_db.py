import numpy as np
from ase.db import connect
from ase.data import covalent_radii,vdw_radii
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from collections import Counter
import sys
sys.path.append('..')
from scripts import metal_colors,metals
from scripts.site_position_functions import get_nearest_sites_in_xy, get_site_pos_in_xy

H_color = np.array([179,227,245])/255
not_fcc=0
energies = []

# Connect to database with atomic structures of *CO
with connect('slabs_out.db') as slab_db,\
    connect('H_out.db') as db,\
    connect('molecules_out.db') as gas_db:

    E_H = 0.5*gas_db._get_row(3).energy
	
	# Iterate through rows in database
    for row in db.select():
		# Check for empty atoms object
        #if row_ads.formula =="":
        #    continue
        #if row_ads.id!=3: continue
		# Get atoms object
        atoms = db.get_atoms(row.id)
		
		# Repeat atoms object
        atoms_3x3 = atoms.repeat((3,3,1))
		
		# Get chemical symbols of atoms
        symbols = np.asarray(atoms_3x3.get_chemical_symbols())
		
		# Get index of central Nitrogen atom
        idx_H= np.nonzero(symbols == 'H')[0][4]
		
		# Get position of hydrogen atom
        pos_H = atoms_3x3.positions[idx_H]
		
		# Get indices of the 1st layer atoms
        ids_1st = np.array([atom.index for atom in atoms_3x3 if atom.tag == 1])
		
        # Get posittion of atoms in 1st layer
        pos_1st=atoms_3x3.positions[ids_1st]
        
		# Get squared distances to the 1st layer atoms from the hydrogen atom
        # dists_sq = np.sum((pos_1st - pos_N)**2, axis=1)
		
		# # Get the closest 3 surface atoms to the hydrogen
		# # (the three-fold hollow adsorption site)
        # ids_site = ids_1st[np.argpartition(dists_sq, kth=2)[:3]]
		
		# # Get the position of the adsorption site as the mean of the positions
		# # of the atoms in the site
        # pos_site = np.mean(atoms_3x3.positions[ids_site], axis=0)
		
		# # Get indices of the 2nd layer atoms
        # ids_2nd = np.array([atom.index for atom in atoms_3x3 if atom.tag == 2])
		
		# # Get squared distances from adsorption site to the atoms in each layer
        # dists_sq_1st = np.sum((atoms_3x3.positions[ids_1st] - pos_site)**2, axis=1)
        # dists_sq_2nd = np.sum((atoms_3x3.positions[ids_2nd] - pos_site)**2, axis=1)
		
        # # Get one atom from 4th and 5th layer 
        # id_4th = np.array([atom.index for atom in atoms_3x3 if atom.tag == 4])[0]
        # id_5th = np.array([atom.index for atom in atoms_3x3 if atom.tag == 5])[0]
        
        # # Get distance in z
        # z_dist = atoms_3x3.positions[id_4th][2] - atoms_3x3.positions[id_5th][2]
        
        # min_dist_2nd = np.sqrt(np.min(dists_sq_2nd))
		
        #print(np.sort(np.sqrt(dists_sq_2nd))[:3],z_dist*1.1)
        #break
        
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
            
        if nearest_site!="fcc":
            not_fcc+=1
            # print(nearest_site,row_ads.id)
            continue
        

        energies.append(row.energy - slab_db.get(slab_idx=row.slab_idx).energy - E_H)


single_metal_energies = []
with connect('single_element_H_out.db') as db, connect('single_element_slabs_out.db') as slab_db:
    for row_ads, row_slab in zip(db.select(),slab_db.select()):
        single_metal_energies.append(row_ads.energy - row_slab.energy - E_H)



print(not_fcc)


fig,ax = plt.subplots(figsize=(6,2))

ax.hist(energies,bins=50,histtype='step',color=H_color)
ax.hist(energies,bins=50,alpha=0.4,color=H_color)

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

# ax.set_xlabel('$E_{*H} - E_{slab}$ (eV)')
ax.set_xlabel('$\Delta E_{*H}$ (eV)')

plt.tight_layout()
plt.savefig('H_ads.png',dpi=600)