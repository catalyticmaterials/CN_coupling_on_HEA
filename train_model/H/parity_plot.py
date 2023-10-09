import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from joblib import load
import itertools as it
import sys
sys.path.append('..')
from scripts import metals, n_metals, metal_colors


ads="H"

#Get regressor
reg = load(f'{ads}/{ads}.joblib')

# Get calculated samples
data = np.loadtxt(f'../features/{ads}.csv', delimiter=',', skiprows=1,usecols=range(46))
ensembles = data[:, :35]
features = data[:,: -1]
energies = data[:, -1]

# Get number of samples
n_samples = len(energies)

# Get predictions of samples
preds = reg.predict(features)

# Get all ensemble combinations
ensembles_ = list(it.combinations_with_replacement(metals, 3))
# Get all ensemble combinations as strings
ensembles_str = ["".join(ens) for ens in ensembles_]

# Assign color to each feature given by the ensemble composition
colors = [metal_colors[ensembles_str[list(ensemble).index(1)]] for ensemble in ensembles] 
#for sample_idx ensemble in enumerate(ensembles):
#	colors.append(metal_colors[ensembles_str[list(ensemble).index(1)]])
	#preds[sample_idx] = reg.predict(tuple(ensemble), feature)
    
    
fig,ax=plt.subplots(figsize=(5,5))

ax.scatter(energies,preds,c=colors)



handles = []
for metal in metals:
    handles.append(Line2D([0], [0], marker='o', color="w", label=metal,markerfacecolor=metal_colors[metal], markersize=10))    


lims=np.array([ax.get_xlim(),ax.get_ylim()]).T
xlim=np.min(lims[0])
ylim=np.max(lims[1])
lim=[xlim,ylim]

ax.plot(lim,lim,c="black",zorder=0)
handles.append(Line2D([0],[0],color="black",label="\u0394$E_{DFT}$=\u0394$E_{pred}$", markersize=10))
plt.legend(handles=handles)
#new lims
ax.set_ylim(lim)
ax.set_xlim(lim)

res=preds - energies
ss_res = np.sum(res**2)
ss_tot=np.sum((energies-np.mean(energies))**2)

R2=np.around(1-ss_res/ss_tot,decimals=3)
MAE = np.mean(np.abs(preds-energies))

plt.text(lim[0]+0.03, 0.05, f"R\u00b2 = {R2}\nMAE = {MAE:0.3f}")

ax.set_xlabel("\u0394$G_{DFT}^{*H}$ [eV]")
ax.set_ylabel("\u0394$G_{pred}^{*H}$ [eV]")

ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.5))


# plt.tight_layout()
plt.subplots_adjust(bottom=0.2,left=0.2)
# plt.savefig("H/H_parity_plot.png",dpi=600,bbox_inches='tight')
plt.show()