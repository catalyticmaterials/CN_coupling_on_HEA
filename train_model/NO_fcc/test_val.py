import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from sklearn.model_selection import LeaveOneOut
import itertools as it
import sys
sys.path.append('..')
from scripts import metals, metal_colors,n_metals
from scripts.regressor import LinearRegressor

# Get ensembles
n_atoms_ensemble = 3
# Get all ensemble combinations
ensembles = list(it.combinations_with_replacement(metals, 3))
# Get all ensemble combinations as strings
ensembles_str = ["".join(ens) for ens in ensembles]

# Get number of ensembles
n_ensembles = len(ensembles)

# Specify zone sizes
n_atoms_zones = [3, 3]
n_zones = len(n_atoms_zones)

# Read features from file
filename = '../features/NO_fcc.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1,usecols=range(46))
features = data[:, :-1]
energies = data[:, -1]

# Define regressor
reg = LinearRegressor(n_ensembles, n_atoms_zones, n_metals)

reg.fit(features,energies)


loo = LeaveOneOut()

preds = []
colors = []
for (train_index,test_index) in loo.split(features):
    reg = LinearRegressor(n_ensembles, n_atoms_zones, n_metals)
    train_features = features[train_index]
    test_feature = features[test_index][0]

    train_energies = energies[train_index]
    test_energy = energies[test_index][0]

    # Train regressor
    reg.fit(train_features, train_energies)



		
    # Predict adsorption energy for feature
    # AE.append(np.abs(reg.predict(tuple(test_feature[:5]),test_feature[5:]) - test_energy))
    
    preds.append(reg.predict(test_feature))

    # colors.append(metal_colors[metals[test_ensemble.index(1)]])

preds = np.array(preds)

# Make parity plot
fig,ax=plt.subplots(figsize=(5,5))

# Assign color to each feature given by the ensemble composition
colors = [metal_colors[ensembles_str[list(ensemble).index(1)]] for ensemble in features[:,:35]]

ax.scatter(energies,preds,c=colors,marker='.')


handles = []
for metal in metals:
    handles.append(Line2D([0], [0], marker='o', color="w", label=metal,markerfacecolor=metal_colors[metal], markersize=10))    


lims=np.array([ax.get_xlim(),ax.get_ylim()]).T
xlim=np.min(lims[0])
ylim=np.max(lims[1])
lim=np.array([xlim,ylim])

ax.plot(lim,lim,c="black",zorder=0)
ax.plot(lim,lim+0.1,c="black",zorder=0,ls=':',alpha=0.5)
line = ax.plot(lim,lim-0.1,c="black",zorder=0,ls=':',alpha=0.5,label="$\pm 0.1$ eV")
handles.append(Line2D([0],[0],color="black",label="\u0394$G_{DFT}$=\u0394$G_{pred}$", markersize=10))
handles.append(line[0])

ax.legend(handles=handles,loc=2)
#new lims
ax.set_ylim(lim)
ax.set_xlim(lim)

res=preds - energies
ss_res = np.sum(res**2)
ss_tot=np.sum((energies-np.mean(energies))**2)

R2=np.around(1-ss_res/ss_tot,decimals=3)

error = preds-energies
MAE = np.mean(np.abs(error))

ax.text(0.45,0.97, f"R\u00b2 = {R2}\nMAE = {MAE:0.3f} eV",va='top',transform=ax.transAxes)

ax.set_xlabel("\u0394$G_{DFT}^{*NO}$ [eV]")
ax.set_ylabel("\u0394$G_{pred}^{*NO}$ [eV]")

ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.5))


ax_inset = fig.add_axes([0.6, 0.3, 0.25, 0.15])

ax_inset.hist(error,color='darkgrey',alpha=0.75)
ylim_ins = ax_inset.get_ylim()
xlim_ins = ax_inset.get_xlim()
ME = np.mean(error)
E_std = np.std(error,ddof=1)
ax_inset.vlines(np.mean(error),*ylim_ins,colors='k',alpha=0.5)
ax_inset.vlines([ME-E_std,ME+E_std],*ylim_ins,colors='k',alpha=0.5,ls='--')
ax_inset.text(ME-E_std,ylim_ins[1],'$-\sigma$',va='bottom',ha='center',fontsize=8)
ax_inset.text(ME+E_std,ylim_ins[1],'$+\sigma$',va='bottom',ha='center',fontsize=8)

# Hide the left, right and top spines
ax_inset.spines[['left','right', 'top']].set_visible(False)
ax_inset.tick_params(left=False)
ax_inset.set_yticks([])
ax_inset.set_xlabel('Residuals [eV]',fontsize=8)
ax_inset.tick_params(labelsize=8)
ax_inset.xaxis.set_minor_locator(MultipleLocator(0.1))

# plt.tight_layout()
plt.subplots_adjust(bottom=0.2,left=0.2)
plt.savefig("NO_fcc/NO_fcc_parity_plot.png",dpi=600,bbox_inches='tight')
# plt.show()

print(np.mean(np.abs(error/energies)))