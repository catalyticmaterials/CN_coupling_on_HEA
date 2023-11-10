import numpy as np
from ase.db import connect
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys
sys.path.append('..')
from scripts import metal_colors,metals


metal_fcc = []
metal_top = []
fcc_energies = []
top_energies = []
slab_energies = []
with connect("single_element_NO_fcc_out.db") as fcc_db, connect("single_element_NO_out.db") as top_db,connect('single_element_slabs_out.db') as slab_db:
    # Iterate through each row of each database
    for (row_fcc,row_top,row_slab) in zip(fcc_db.select(),top_db.select(),slab_db.select()):
        metal_fcc.append(row_fcc.metal)
        metal_top.append(row_top.metal)
        
        fcc_energies.append(row_fcc.energy)
        top_energies.append(row_top.energy)
        slab_energies.append(row_slab.energy)

slab_energies = np.array(slab_energies)
# Get soring ids so that metals are in same order
sort_fcc = np.argsort(metal_fcc)
sort_top = np.argsort(metal_top)

fcc_energies=np.array(fcc_energies)[sort_fcc] - slab_energies
top_energies=np.array(top_energies)[sort_top] - slab_energies


# Get metal colors
single_metal_colors = [metal_colors[metal] for metal in metals]

# Make plot
fig,ax = plt.subplots(figsize=(4,4))
plt.scatter(fcc_energies,top_energies,color=single_metal_colors,zorder=1)

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
plt.savefig('NO_fcc_vs_top.png',dpi=600,bbox_inches='tight')