import numpy as np
from ase.db import connect
from ase.data import covalent_radii,vdw_radii
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from collections import Counter
import sys
sys.path.append('..')
from scripts import metal_colors,metals


# def get_site_pos_in_xy(pos_1st):
#     # Rearange positions so that they are ordered by layer
#     pos_1st_ = np.array([pos_1st[(9*3*i)+j*3:3+3*9*i+j*3] for j in range(9) for i in range(3)])
#     pos_1st_ = pos_1st_.reshape(81,3)
#     #print(pos_1st_)
#     grid = pos_1st_.reshape(9,9,-1)
#     #print(grid)
#     grid = np.pad(grid,pad_width=((1,1),(1,1),(0,0)),mode="wrap")
    
#     fcc_sites =  (grid[1:-1,1:-1] + grid[1:-1,2:] + grid[2:,1:-1])/3
    
#     hcp_sites = (grid[1:-1,1:-1] + grid[2:,:-2] +grid[2:,1:-1])/3
    
#     bridge_sites1 = (grid[1:-1,1:-1] + grid[1:-1,2:])/2
#     bridge_sites2 = (grid[1:-1,1:-1] + grid[2:,1:-1])/2
#     bridge_sites3 = (grid[1:-1,1:-1] + grid[2:,:-2])/2
    
    
#     # cut off ends as we are only interested in sites in the middle 3x3 atoms anyway
#     fcc_sites = fcc_sites[2:-2,2:-2]
#     hcp_sites = fcc_sites[2:-2,2:-2]
#     bridge_sites1 = bridge_sites1[2:-2,2:-2]
#     bridge_sites2 = bridge_sites2[2:-2,2:-2]
#     bridge_sites3 = bridge_sites3[2:-2,2:-2]
#     ontop_sites = np.copy(grid[3:-3,3:-3]).reshape(-1,3)
    
#     bridge_sites = np.vstack([bridge_sites1.reshape(-1,3),bridge_sites2.reshape(-1,3),bridge_sites3.reshape(-1,3)])
#     return fcc_sites.reshape(-1,3), hcp_sites.reshape(-1,3), bridge_sites,ontop_sites

# def get_nearest_sites_in_xy(fcc,hcp,bridge,ontop,ads):
#     fcc_dist = np.sum((fcc[:,:2]-ads[:2])**2,axis=1)
#     hcp_dist = np.sum((hcp[:,:2]-ads[:2])**2,axis=1)
#     bridge_dist = np.sum((bridge[:,:2]-ads[:2])**2,axis=1)
#     ontop_dist = np.sum((ontop[:,:2]-ads[:2])**2,axis=1)
    
#     min_ids = [np.argmin(dist) for dist in (fcc_dist,hcp_dist,bridge_dist,ontop_dist)]
#     min_dists = [dist[min_ids[i]] for i,dist in enumerate((fcc_dist,hcp_dist,bridge_dist,ontop_dist))]
    
#     site_str = ["fcc","hcp","bridge","ontop"]
    
#     nearest_site_type = site_str[np.argmin(min_dists)]
    
#     return nearest_site_type, min_ids

# def return_nearest_site(row,db):
#     # Get atoms object
#     atoms=db.get_atoms(row.id)
    
#     # Repeat atoms object
#     atoms_3x3 = atoms.repeat((3,3,1))

#     #Get chemical symbols of atoms
#     symbols = np.asarray(atoms_3x3.get_chemical_symbols())

#     #Get index of central hydrogen atom
#     idx_N = np.nonzero(symbols == 'N')[0][4]

#     #Get position of carbon atom
#     pos_N = atoms_3x3.positions[idx_N]

#     #Get indices of the 1st layer atoms
#     ids_1st = np.array([atom.index for atom in atoms_3x3 if atom.tag == 1])

#     # Get posittion of atoms in 1st layer
#     pos_1st=atoms_3x3.positions[ids_1st]
    
#     #Get all site positions
#     fcc_sites_pos,hcp_sites_pos,bridge_sites_pos,ontop_sites_pos = get_site_pos_in_xy(pos_1st)
    
#     nearest_site,min_dist_ids=get_nearest_sites_in_xy(fcc_sites_pos, hcp_sites_pos, bridge_sites_pos, ontop_sites_pos, pos_N)
#     return nearest_site, pos_N

# fcc_energies = []
# top_energies = []
# fcc_sites = []
# top_sites = []
# pos_diff = []

# with connect("NO_fcc_out.db") as fcc_db, connect("NO_out_sorted.db") as top_db:
#     for (row_fcc,row_top) in zip(fcc_db.select(),top_db.select()):
#         if row_fcc.formula=="":
#             continue
#         fcc_energies.append(row_fcc.energy)
#         top_energies.append(row_top.energy)
        
#         nearest_site_fcc,pos_N_fcc = return_nearest_site(row_fcc, fcc_db)
#         nearest_site_top,pos_N_top = return_nearest_site(row_top, top_db)
        
#         fcc_sites.append(nearest_site_fcc)
#         top_sites.append(nearest_site_top)
        
#         pos_diff.append(np.linalg.norm(pos_N_fcc-pos_N_top))
        
        
        
# energy_diff =  np.array(fcc_energies)-np.array(top_energies)
# plt.figure(dpi=400)
# plt.scatter(fcc_energies,energy_diff,label="Data points",marker=".")
# plt.plot(fcc_energies,np.zeros(len(fcc_energies)),c="k")
# plt.xlabel("E$_{NO,FCC}$ [eV]")
# plt.ylabel("E$_{NO,FCC} - E_{NO,top}$ [eV]")
# plt.legend()


# plt.figure(dpi=400)
# plt.hist(energy_diff,histtype="step",label="E$_{NO,FCC} - E_{NO,top}$",bins=20)
# plt.text(-1.8,90,f"Mean difference = {np.mean(energy_diff):.2f} eV")
# plt.legend(loc=2)
# plt.xlabel("Energy difference [eV]")
# plt.ylabel("Counts")

# plt.figure()
# plt.hist(fcc_energies,histtype="step",bins=100)
# plt.hist(top_energies,histtype="step",bins=100)
# print(np.mean(fcc_energies),np.mean(top_energies))


# print(Counter(fcc_sites))
# print(Counter(top_sites))
# plt.figure(dpi=400)
# top_counter =  Counter(top_sites)
# plt.bar(x=1,height=top_counter["ontop"],label="top")
# plt.bar(x=2,height=top_counter["bridge"],label="bridge")
# plt.bar(x=3,height=top_counter["fcc"],label="fcc")
# plt.xticks([1,2,3],labels=["top","bridge","fcc"])
# plt.ylabel("Number of sites")

# plt.figure(dpi=400)
# fcc_counter =  Counter(fcc_sites)
# plt.bar(x=1,height=fcc_counter["ontop"],label="top")
# plt.bar(x=2,height=fcc_counter["bridge"],label="bridge")
# plt.bar(x=3,height=fcc_counter["fcc"],label="fcc")
# plt.xticks([1,2,3],labels=["top","bridge","fcc"])
# plt.ylabel("Number of sites")


# plt.figure()
# plt.hist(pos_diff,bins=30)
# print(np.mean(pos_diff))


metal_fcc = []
metal_top = []
fcc_energies = []
top_energies = []
slab_energies = []
with connect("single_element_NO_fcc_out.db") as fcc_db, connect("single_element_NO_angled_out.db") as top_db,connect('single_element_slabs_out.db') as slab_db:
    for (row_fcc,row_top,row_slab) in zip(fcc_db.select(),top_db.select(),slab_db.select()):
        metal_fcc.append(row_fcc.metal)
        metal_top.append(row_top.metal)
        
        fcc_energies.append(row_fcc.energy)
        top_energies.append(row_top.energy)
        slab_energies.append(row_slab.energy)

slab_energies = np.array(slab_energies)

sort_fcc = np.argsort(metal_fcc)
sort_top = np.argsort(metal_top)

fcc_energies=np.array(fcc_energies)[sort_fcc] - slab_energies
top_energies=np.array(top_energies)[sort_top] - slab_energies




"""
plt.figure()
plt.scatter(fcc_energies,top_energies)
plt.plot(fcc_energies,fcc_energies,c="k")

for i,metal in enumerate(metals):
    plt.text(fcc_energies[i]+0.1,top_energies[i]+0.1,s=metal)
    
print(fcc_energies-top_energies)
"""
energy_diff =  np.array(fcc_energies)-np.array(top_energies)

single_metal_colors = [metal_colors[metal] for metal in metals]

fig,ax = plt.subplots(figsize=(4,4))
plt.scatter(fcc_energies,top_energies,color=single_metal_colors,zorder=1)

# plt.plot(fcc_energies,np.zeros(len(fcc_energies)),c="k")
xlim = ax.get_xlim()
ylim = ax.get_ylim()
plt.plot(xlim,xlim,c='k',ls=':',zorder=0)

plt.xlabel('$E_{*NO}^{fcc} - E_{slab}$ (eV)')
plt.ylabel('$E_{*NO}^{top} - E_{slab}$ (eV)')


for i,metal in enumerate(metals):
    plt.text(fcc_energies[i],top_energies[i]+0.02,s=metal,color=single_metal_colors[i],va='bottom',ha='center')

ax.set_ylim(*xlim)
ax.set_xlim(*xlim)

ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MultipleLocator(0.5))

ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

plt.subplots_adjust(bottom=0.2,left=0.2)

#plt.tight_layout()

# plt.figure(dpi=400)
# plt.hist(energy_diff,histtype="step",label="E$_{NO,FCC} - E_{NO,top}$",bins=20)
# #plt.text(-1.8,90,f"Mean difference = {np.mean(energy_diff):.2f} eV")
# plt.legend(loc=2)
# plt.xlabel("Energy difference [eV]")
# plt.ylabel("Counts")

plt.savefig('NO_fcc_vs_top.png',dpi=600,bbox_inches='tight')