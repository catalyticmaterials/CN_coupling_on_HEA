import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel

import os
import sys
sys.path.append("..")
from shared_params.surface import *
from shared_params.compositionspace_functions import molar_fractions_to_cartesians
from shared_params.bayesian_sampling import BayesianSampler
from shared_params.gaussian_process_regressor import GPR


metals = ['Ag','Au', 'Cu', 'Pd','Pt']



#Set up gpr
gpr=GPR(kernel=C(1.0) * RBF(1.0) + WhiteKernel(noise_level=0.001,noise_level_bounds=(1e-20,1)),n_restarts_optimizer=25,alpha=0,max_iter=2e+6,gtol=1e-12)

BS=BayesianSampler(n_elems=5)

np.random.seed(1)

def predict_prob(composition):

    surface = initiate_surface(composition,metals)
    
    NO_energies, site_ids= predict_energies(surface, "NO", metals)
    CO_energies, site_ids= predict_energies(surface, "CO", metals)
    
    NO_energies += -0.71
    CO_energies += -0.4
        
    mask_x = lambda x: (x>=(-1.1))*(x<=(-0.4))
    mask_y = lambda y: (y>=(-1.3))*(y<=(-0.71))
    
    prob=(sum(mask_x(CO_energies))/10000) * (sum(mask_y(NO_energies))/10000)
    return prob


f_train = np.array([])
probs= f_train.copy()

f_train= BS.get_molar_fraction_samples(f_train, probs, gpr)
probs = np.array([predict_prob(f) for f in f_train])
r_train=molar_fractions_to_cartesians(f_train)
print(r_train)
gpr.fit(r_train,probs)


for i in range(10):
    f_train= BS.get_molar_fraction_samples(f_train, probs, gpr)
    f_next = f_train[-1]
    probs = np.append(probs,predict_prob(f_next))
    r_train=molar_fractions_to_cartesians(f_train)
    gpr.fit(r_train,probs)
    print(f_next)

print(np.round(f_train,decimals=3))
print(np.round(probs,decimals=3)) 