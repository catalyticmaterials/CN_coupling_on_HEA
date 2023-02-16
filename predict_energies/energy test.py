import numpy as np
from shared_params.surface import *
import matplotlib.pyplot as plt
np.random.seed(1)
metals = ['Ag','Au', 'Cu', 'Pd','Pt']
surface = initiate_surface(np.array([0.2,0.2,0.2,0.2,0.2]),metals)
energies=predict_energies(surface, "H", metals)

plt.hist(energies,bins=int(np.sqrt(len(energies))))
