import numpy as np
import os
import sys
sys.path.append("..")
from shared_params.surface import *


metals = ['Ag','Au', 'Cu', 'Pd','Pt']
composition = np.array([0.1,0.1,0.6,0.1,0.1])

alloy_list = [metals[i]+str(composition[i]) for i in range(len(metals))]
alloy = "".join(alloy_list)

if os.path.isdir(f"{alloy}")==False:
    os.makedirs(f"{alloy}")

surface = initiate_surface(composition,metals)

for ads in ["CO","H","NO"]:
    np.random.seed(1)
    energies,site_ids=predict_energies(surface, ads, metals)
    np.savetxt(f"{alloy}/{ads}_energies.csv", np.hstack((energies.reshape(-1,1),site_ids)))


