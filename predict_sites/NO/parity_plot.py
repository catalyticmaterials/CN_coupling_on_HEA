import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from joblib import load
import sys
sys.path.append('../..')
from shared_params import metals, n_metals, metal_colors

ads="NO"

#Get regressor
reg = load(f'{ads}.joblib')

# Get calculated samples
data = np.loadtxt(f'../../features/{ads}.csv', delimiter=',', skiprows=1)
ensembles = data[:, :5]
features = data[:, 5 : -3]
energies = data[:, -3]

# Get number of samples
n_samples = len(energies)

# Get predictions of samples
preds = np.zeros(n_samples)
colors = [] 
for sample_idx, (ensemble, feature) in enumerate(zip(ensembles, features)):
	colors.append(metal_colors[metals[list(ensemble).index(1)]])
	preds[sample_idx] = reg.predict(tuple(ensemble), feature)
    
    
fig,ax=plt.subplots(dpi=400)

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

plt.text(lim[0]+0.05, -0.9, f"R\u00b2 = {R2}")

ax.set_xlabel("\u0394$E_{DFT}^{*NO}$ relative to \u0394$E_{Cu(111)}^{*NO}$ [eV]")
ax.set_ylabel("\u0394$E_{pred}^{*NO}$ relative to \u0394$E_{Cu(111)}^{*NO}$ [eV]")
