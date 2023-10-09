import numpy as np
from ase.db import connect
from ase.data import covalent_radii,vdw_radii
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys
sys.path.append('..')
from scripts.site_position_functions import get_nearest_sites_in_xy, get_site_pos_in_xy

import sys
sys.path.append('..')
from scripts import metal_colors,metals

not_ontop=0
energies = []
site_metals = []

dG_CO_0 = -0.4
dG_CO_on_Cu = -0.12

with connect('single_element_CO_out.db') as db_ads, connect('single_element_slabs_out.db') as db_slabs, connect('molecules_out.db') as db_gas:
    E_CO = db_gas._get_row(2).energy 
    E_ref = db_ads._get_row(3).energy - db_slabs._get_row(3).energy - E_CO - dG_CO_0 - dG_CO_on_Cu


with connect("CO_out.db") as db, connect('slabs_out.db') as slab_db:
    for row in db.select():
        atoms=db.get_atoms(row.id)
        site_idx=row.site_idx
        
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
		
        # # Get the three shortest distances
        # dist_3 = np.sort(np.sqrt(dists_sq))[:3]  
        
        # # Get one atom from 4th and 5th layer 
        # id_4th = np.array([atom.index for atom in atoms_3x3 if atom.tag == 4])[0]
        # id_5th = np.array([atom.index for atom in atoms_3x3 if atom.tag == 5])[7]
        
        # # Get distance in z
        # z_dist = atoms_3x3.positions[id_4th][2] - atoms_3x3.positions[id_5th][2]
        
        # # Set average radius of atoms
        # r = z_dist/2
        # # Get radius of N
        # r_C = covalent_radii[6]
        
        # #Set reference distance
        # r_ref = r + r_C
        
        # #Get number of distances below reference +20%
        # n = np.sum(dist_3 < r_ref*1.2)

        # if n>1:
        #     not_ontop+=1
        #     print(row.id)

        
        




        # Get posittion of atoms in 1st layer
        pos_1st=atoms_3x3.positions[ids_1st]
        #Get position of each site type in xy-plane from the 1st layer atom position
        sites_xy = get_site_pos_in_xy(pos_1st)
        fcc_sites_pos, hcp_sites_pos, bridge_sites_pos, ontop_sites_pos = sites_xy
        #Get site type and min dist site id.
        nearest_site, min_dist_ids = get_nearest_sites_in_xy(*sites_xy,pos_C)
        
        # If nearest site is bridge, locate the next nearest site
        if nearest_site=="bridge": 
            fcc_dist = np.linalg.norm(fcc_sites_pos[min_dist_ids[0]][:2] - pos_C[:2])
            hcp_dist =np.linalg.norm(hcp_sites_pos[min_dist_ids[1]][:2] - pos_C[:2])
            top_dist =np.linalg.norm(ontop_sites_pos[min_dist_ids[3]][:2] - pos_C[:2])
            site_names = ["fcc","hcp","ontop"]
            nearest_site = site_names[np.argmin([fcc_dist,hcp_dist,top_dist])]
            
        if nearest_site!="ontop":
            not_ontop+=1
            print(nearest_site,row.id)
            continue


        # Get the closest surface atom to the carbon (the on-top adsorption site)
        idx_site = ids_1st[np.argmin(dists_sq)]		

        # Get symbol of adsorption site
        symbol_site = symbols[idx_site]

        site_metals.append(symbol_site)

        energies.append(row.energy - slab_db.get(slab_idx=row.slab_idx).energy - E_CO)

        

#print(ontop,bridge,hollow)
#print(sum([ontop,bridge,hollow]))
print("not ontop:",not_ontop)

site_metals = np.array(site_metals)
energies = np.array(energies) - E_ref

fig,ax = plt.subplots(figsize=(6,2))

for metal in metals:
    mask = site_metals==metal
    ax.hist(energies[mask],bins=60,range=(np.min(energies),np.max(energies)),color=metal_colors[metal],histtype='step')
    ax.hist(energies[mask],bins=60,range=(np.min(energies),np.max(energies)),color=metal_colors[metal],alpha=0.4,label=metal)


ax.set_xlabel('$\Delta E_{*CO}$ (eV)')
# ax.set_xlabel('$E_{*CO} - E_{slab}$ (eV)')
# ax.set_ylabel('Counts')
ax.set_ylim(None,52)

# Hide the left, right and top spines
ax.spines[['left','right', 'top']].set_visible(False)
ax.tick_params(left=False)
ax.set_yticks([])
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MultipleLocator(0.5))
# import matplotlib.patches as mpatches

handles, labels = ax.get_legend_handles_labels()
#handles = [handles[4], mpatches.Patch(color='none'), handles[3], mpatches.Patch(color='none'), handles[2],mpatches.Patch(color='none'),handles[1],handles[0]]
#labels = [labels[4],'',labels[3],'',labels[2],'',labels[1],labels[0]]


ax.legend(handles, labels,loc=1,frameon=False,ncol=5,mode='expand',reverse=True)

plt.tight_layout()
plt.savefig('CO_ads.png',dpi=600)

