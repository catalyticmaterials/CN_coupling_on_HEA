import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from tqdm import tqdm
import sys
sys.path.append("..")
from scripts.compositionspace_functions import molar_fractions_to_cartesians
from scripts.bayesian_sampling import BayesianSampler
from scripts.methods import count_sites

metals = ['Ag','Au', 'Cu', 'Pd','Pt']


for P_NO in [1,0.5,0.1]:

    #Set up gpr and Bayes sampler classes
    gpr=GPR(kernel=C(1.0) * RBF(1.0) + WhiteKernel(noise_level=0.001,noise_level_bounds=(1e-20,1)),n_restarts_optimizer=25,alpha=0)#,max_iter=2e+6)#,gtol=1e-12)
    BS=BayesianSampler(n_elems=5)

    # Set a seed for reproduceability
    np.random.seed(42)


    # Initiate arrays
    f_train = np.array([])
    n_sites = f_train.copy()

    # Get initial samples and their number of sites
    f_train= BS.get_molar_fraction_samples(f_train, n_sites, gpr)
    n_sites = np.array([count_sites(f, P_CO=1.0, P_NO=P_NO,metals=metals,method='mc') for f in f_train])
    r_train=molar_fractions_to_cartesians(f_train)

    # Train gpr on initial samples
    gpr.fit(r_train,n_sites)

    # Bayesian optimization
    for i in tqdm(range(200)):
        # Update training list and get next sample
        f_train= BS.get_molar_fraction_samples(f_train, n_sites, gpr)
        f_next = f_train[-1]

        # If next molar fraction is near 1 for one element, get pure metal instead
        if np.any(f_next>=0.993749):
            mask = f_next>=0.993749
            f_next = np.zeros(5)
            f_next[mask] = 1
            f_train[-1] = f_next

        # Get number of sites for new sample and append result
        n_sites = np.append(n_sites,count_sites(f_next, P_CO=1.0, P_NO=P_NO,metals=metals,method='mc'))
        r_train=molar_fractions_to_cartesians(f_train)
        # Update gpr
        gpr.fit(r_train,n_sites)

    P_NO_str = f'{P_NO}'.replace('.','')
    # Save data
    data = np.hstack((f_train,n_sites.reshape(-1,1)))
    np.savetxt(f'MC/Bayesian_optimization_results_PNO_{P_NO_str}.csv',data,delimiter=',',fmt=['%.2f','%.2f','%.2f','%.2f','%.2f','%.4f'],header='Ag,Au,Cu,Pd,Pt,active sites')
