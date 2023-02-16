import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from tqdm import tqdm
import os
import sys
sys.path.append("..")
from shared_params.surface import predict_energies, initiate_surface
from shared_params.compositionspace_functions import get_molar_fractions
from shared_params import metal_colors

metals = ['Ag','Au', 'Cu', 'Pd','Pt']



def prob_densities(NO_energies,CO_energies):
    NO_energies += -0.71
    CO_energies += -0.4
    
    
    mask_x = lambda x: (x>=(-1.1))*(x<=(-0.4))
    mask_y = lambda y: (y>=(-1.3))*(y<=(-0.71))
    
    prob_urea=(sum(mask_x(CO_energies))/10000) * (sum(mask_y(NO_energies))/10000)
    
    
    mask_Hx = lambda x: x<(-1.1)
    mask_Hy = lambda y: y<(-1.3)
    
    prob_H = sum(mask_Hx(CO_energies) * mask_Hy(NO_energies))/10000
    return prob_urea,prob_H


probs_urea = np.empty(0)
probs_H = np.empty(0)

grid = get_molar_fractions(0.25, 5)

for mf in tqdm(grid):
    surface = initiate_surface(mf,metals)
    
    NO_energies, site_ids= predict_energies(surface, "NO_fcc", metals)
    CO_energies, site_ids= predict_energies(surface, "CO", metals)
    
    prob_urea,prob_H = prob_densities(NO_energies, CO_energies)
    probs_urea = np.append(probs_urea,prob_urea)
    probs_H = np.append(probs_H,prob_H)



data = np.hstack((grid,probs_urea.reshape(-1,1),probs_H.reshape(-1,1)))
np.savetxt("probabilities.csv", data,delimiter=",")



colors = [np.sum([np.array(to_rgb(metal_colors[metal]))*f for metal,f in zip(metals,mf)],axis=0) for mf in grid]
    
plt.figure(dpi=400)
plt.scatter(probs_H*100,probs_urea*100,c=colors)

plt.xlabel("Sites allowing H$_2$ formation [%]")
plt.ylabel("Sites allowing Urea formation [%]")
handles = [Line2D([0], [0], marker='o', color="w", label=metal,markerfacecolor=metal_colors[metal], markersize=10) for metal in metals]
plt.legend(handles=handles)