import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from sklearn.model_selection import LeaveOneOut
import sys
sys.path.append('..')
from scripts import metals, metal_colors
from scripts.regressor import MultiLinearRegressor

# Specify regressor
n_atoms_zones = [6, 3, 3]
n_zones = len(n_atoms_zones)
reg = MultiLinearRegressor(n_atoms_zones)

# Read features from file
filename = '../features/CO.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)
features = data[:, :-4]
energies = data[:, -4]


loo = LeaveOneOut()

preds = []
colors = []
for (train_index,test_index) in loo.split(features):
	
    train_features = features[train_index]
    test_feature = features[test_index][0]
	
    train_energies = energies[train_index]
    test_energy = energies[test_index][0]
	
    

    # Split features and energies into ensembles
    ensembles = [(1,0,0,0,0), (0,1,0,0,0), (0,0,1,0,0), (0,0,0,1,0), (0,0,0,0,1)]
    for ensemble in ensembles:
        
        # Only use those features that start with the current ensemble
        mask = np.all(train_features[:, :5] == ensemble, axis=1)

        # Train regressor on this ensemble
        reg.fit(ensemble, train_features[mask, 5:], train_energies[mask])

		
    # Predict adsorption energy for feature
    # AE.append(np.abs(reg.predict(tuple(test_feature[:5]),test_feature[5:]) - test_energy))
    test_ensemble = tuple(test_feature[:5])
    preds.append(reg.predict(test_ensemble,test_feature[5:]))

    colors.append(metal_colors[metals[test_ensemble.index(1)]])



preds = np.array(preds)

# Make parity plot
fig,ax=plt.subplots(figsize=(5,5))

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

plt.legend(handles=handles)
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

ax.set_xlabel("\u0394$G_{DFT}^{*CO}$ [eV]")
ax.set_ylabel("\u0394$G_{pred}^{*CO}$ [eV]")

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

# plt.tight_layout()
plt.subplots_adjust(bottom=0.2,left=0.2)
plt.savefig("CO/CO_parity_plot.png",dpi=600,bbox_inches='tight')

print(np.mean(np.abs(error/energies)))