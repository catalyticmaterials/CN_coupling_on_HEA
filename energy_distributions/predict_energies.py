import numpy as np
import os
import sys
sys.path.append("..")
from scripts.surface import *


metals = ['Ag','Au', 'Cu', 'Pd','Pt']

# Loop through selected compositions
for composition in [np.array([0.2,0.2,0.2,0.2,0.2]),np.array([0.1,0.1,0.6,0.1,0.1]),np.array([0.0,0.5,0.5,0.0,0.0])]:
    # Set the same seed for every composition
    np.random.seed(1)

    # Make a string og the alloy
    alloy_list = [metals[i] + str(int(composition[i]*100)) for i in range(len(metals)) if composition[i]>0]
    alloy = "".join(alloy_list)

    # Make directory of the alloy name
    if os.path.isdir(f"{alloy}")==False:
        os.makedirs(f"{alloy}")

    # Simulate the surface
    surface = initiate_surface(composition,metals)
    
    # Predict energies of each adsorbate
    for ads in ["CO","NO_fcc",'H']:
        energies,site_ids=predict_energies(surface, ads, metals)
        np.savetxt(f"{alloy}/{ads}_energies.csv", np.hstack((energies.reshape(-1,1),site_ids)),delimiter=',')
        #print(np.sort(-energies)[:5])
        
    