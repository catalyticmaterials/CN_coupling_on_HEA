import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress
from matplotlib.ticker import MultipleLocator
import sys
sys.path.append("..")
from scripts.surface import predict_energies, initiate_surface
from scripts import metals, metal_colors

# Set random seed
np.random.seed(1)

# Set composition
composition = np.array([0.2,0.2,0.2,0.2,0.2])

# Initiate HEA surface
surface = initiate_surface(composition,metals)

# Predict adsorption energies
CO_energies, site_ids = predict_energies(surface,'CO',metals)
NO_energies, site_ids = predict_energies(surface,'NO_fcc',metals)
H_energies, site_ids = predict_energies(surface,'H',metals)

# Correct to adsorption energies
NO_energies-=0.71
CO_energies-=0.4
H_energies-=0.1

# Get top layer surface
top_layer = surface[:,:,0]

# Turn energies back into grids
# CO_grid = CO_energies.reshape(100,100)
NO_grid = NO_energies.reshape(100,100)
H_grid = H_energies.reshape(100,100)

# Get H and NO energies around CO:
# First, make padded NO and H grid
NO_grid_pad = np.pad(NO_grid,pad_width=((1,1),(1,1)),mode='wrap')
H_grid_pad = np.pad(H_grid,pad_width=((1,1),(1,1)),mode='wrap')

# Get surrounding fcc site energies of top sites 
NO_energies_around_top = np.hstack((NO_grid_pad[1:-1,1:-1].reshape(-1,1),NO_grid_pad[1:-1,:-2].reshape(-1,1),NO_grid_pad[:-2,1:-1].reshape(-1,1)))
H_energies_around_top = np.hstack((H_grid_pad[1:-1,1:-1].reshape(-1,1),H_grid_pad[1:-1,:-2].reshape(-1,1),H_grid_pad[:-2,1:-1].reshape(-1,1)))



# Prepare colors
top_colors = np.array([metal_colors[metal] for metal in top_layer.flatten()])

top_layer_pad = np.pad(top_layer,pad_width=((1,1),(1,1)),mode='wrap')
fcc_sites = np.hstack((top_layer_pad[1:-1,1:-1].reshape(-1,1),top_layer_pad[1:-1,:-2].reshape(-1,1),top_layer_pad[:-2,1:-1].reshape(-1,1)))
fcc_colors = np.array([metal_colors[''.join(np.sort(fcc_metals))] for fcc_metals in fcc_sites])


fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(13,4))

for i in range(3):
    ax1.scatter(H_energies_around_top[:,i][H_energies_around_top[:,i]<10],CO_energies[H_energies_around_top[:,i]<10],marker='.',c=top_colors[H_energies_around_top[:,i]<10])
    ax3.scatter(CO_energies,NO_energies_around_top[:,i],marker='.',c=top_colors)

ax2.scatter(H_energies[H_energies<10],NO_energies[H_energies<10],marker='.',c=fcc_colors[H_energies<10])


a,b, *ignore= linregress(H_energies[H_energies<10],NO_energies[H_energies<10])

xlim1, ylim1 = ax1.get_xlim(), ax1.get_ylim()
xlim2, ylim2 = ax2.get_xlim(), ax2.get_ylim()
xlim3, ylim3 = ax3.get_xlim(), ax3.get_ylim()


ax1.vlines(-0.1, ylim1[0],ylim1[1],color='k',ls='--',alpha=0.6)
ax1.hlines(-0.4, xlim1[0],xlim1[1],color='k',ls='--',alpha=0.6)
ax1.set_xlim(*xlim1)
ax1.set_ylim(*ylim1)


xarr = np.array([xlim2[0],xlim2[1]])
ax2.plot(xarr,a*xarr + b,label='Fit',c='b',ls='--')

ax2.vlines(-0.1, ylim2[0],ylim2[1],color='k',ls='--',alpha=0.6)
ax2.hlines(-0.71, xlim2[0],xlim2[1],color='k',ls='--',alpha=0.6)
ax2.set_xlim(*xlim2)
ax2.set_ylim(*ylim2)

ax2.text(-0.1+0.04,-1.65,'\u0394$E_{pred}^{*NO} = $' + f'{a:.2f}' + '\u0394$E_{pred}^{*H}$' + f'\u2212{abs(b):.2f} eV',fontsize=9,color='b')
ax2.legend(loc=2)

ax3.vlines(-0.4, ylim3[0],ylim3[1],color='k',ls='--',alpha=0.6)
ax3.hlines(-0.71, xlim3[0],xlim3[1],color='k',ls='--',alpha=0.6)
ax3.hlines(a*(-0.1)+b, xlim3[0],xlim3[1],color='b',ls='--')
ax3.set_xlim(*xlim3)
ax3.set_ylim(*ylim3)

ax1.set_xlabel("\u0394$E_{pred}^{*H}$ [eV]")
ax1.set_ylabel("\u0394$E_{pred}^{*CO}$ [eV]")

ax2.set_xlabel("\u0394$E_{pred}^{*H}$ [eV]")
ax2.set_ylabel("\u0394$E_{pred}^{*NO}$ [eV]")

ax3.set_xlabel("\u0394$E_{pred}^{*CO}$ [eV]")
ax3.set_ylabel("\u0394$E_{pred}^{*NO}$ [eV]")

for i,ax in enumerate([ax1,ax2,ax3]):
    if i==2:
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
  



handles = []
for metal in metals:
    handles.append(Line2D([0], [0], marker='o', color="w", label=metal,markerfacecolor=metal_colors[metal], markersize=10)) 

#plt.tight_layout()
#

fig.legend(handles=handles, labels=metals,
           loc='outside upper center', ncol=5, mode='expand',fontsize=10,bbox_to_anchor=(0.063, .5, 0.925, 0.5),fancybox=False)

plt.tight_layout()
fig.subplots_adjust(top=0.9)
plt.savefig('scaling.png', dpi=400)
