from .colors import metal_colors
#from .lattice_parameters import lattice_parameters
from scipy import special
import numpy as np
import itertools as it

# Specify metals in alloy
metals = ['Ag', 'Au', 'Cu', 'Pd', 'Pt']
n_metals = len(metals)

G_corr = {
    'NO': 0.56,
    'CO': 0.40,
    'H': 0.2
}

def count_elements(elements, n_elems):
	count = np.zeros(n_elems, dtype=int)
	for elem in elements:
		count[elem] += 1
	return count

def get_molar_fractions(step_size, n_elems, total=1., return_number_of_molar_fractions=False):
	'Get all molar fractions with the given step size'
	
	interval = int(total/step_size)
	n_combs = special.comb(n_elems+interval-1, interval, exact=True)
	
	if return_number_of_molar_fractions:
		return n_combs
		
	counts = np.zeros((n_combs, n_elems), dtype=int)

	for i, comb in enumerate(it.combinations_with_replacement(range(n_elems), interval)):
		counts[i] = count_elements(comb, n_elems)

	return counts*step_size
