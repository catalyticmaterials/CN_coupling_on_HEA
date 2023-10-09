import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from joblib import load
import sys
sys.path.append('..')
from scripts import metals, n_metals, metal_colors

ads="CO"

#Get regressor
reg = load(f'{ads}/{ads}.joblib')

# Get calculated samples
data = np.loadtxt(f'../features/{ads}.csv', delimiter=',', skiprows=1)
ensembles = data[:, :5]
features = data[:, 5 : -4]
energies = data[:, -4]

# Get number of samples
n_samples = len(energies)

# Get predictions of samples
preds = np.zeros(n_samples)
colors = [] 
for sample_idx, (ensemble, feature) in enumerate(zip(ensembles, features)):
	colors.append(metal_colors[metals[list(ensemble).index(1)]])
	preds[sample_idx] = reg.predict(tuple(ensemble), feature)
    
    
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
print(np.abs(preds-energies)[:5])
MAE = np.mean(np.abs(preds-energies))

plt.text(lim[0]+0.03, -0.7, f"R\u00b2 = {R2}\nMAE = {MAE:0.3f}")

ax.set_xlabel("\u0394$G_{DFT}^{*CO}$ [eV]")
ax.set_ylabel("\u0394$G_{pred}^{*CO}$ [eV]")

ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.5))


# plt.tight_layout()
plt.subplots_adjust(bottom=0.2,left=0.2)
# plt.savefig("{ads}/CO_parity_plot.png",dpi=600,bbox_inches='tight')
plt.show()