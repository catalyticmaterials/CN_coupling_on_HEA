import numpy as np
from joblib import dump
import itertools as it
from collections import Counter
from math import factorial as fac
import sys
sys.path.append('../..')
from shared_params import metals, n_metals
from shared_params.regressor import MultiLinearRegressor

# Specify regressor
n_atoms_zones = [6, 3, 3]
n_zones = len(n_atoms_zones)
reg = MultiLinearRegressor(n_atoms_zones)

# Read features from file
filename = '../../features/NO.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)
features = data[:, :-3]
energies = data[:, -3]

# Split features and energies into ensembles
ensembles = [(1,0,0,0,0), (0,1,0,0,0), (0,0,1,0,0), (0,0,0,1,0), (0,0,0,0,1)]
for ensemble in ensembles:
	
	# Only use those features that start with the current ensemble
	mask = np.all(features[:, :5] == ensemble, axis=1)

	# Train regressor on this ensemble
	reg.fit(ensemble, features[mask, 5:], energies[mask])

# Save linear parameters to file
intercepts, slopes = reg.get_parameters()
for metal, ens, intercept, slopes_ in zip(metals, ensembles, intercepts.values(), slopes.values()):
	
	filename = f'NO_{metal}_linear_parameters.csv'
	with open(filename, 'w') as file_:
		
		file_.write(f'intercept,{intercept:>10.6f}')
		
		for zone_idx, zone in enumerate(['1a', '2a', '3b']):
		
			for slope, metal in zip(slopes_[zone_idx*n_metals : (zone_idx+1)*n_metals], metals):
				file_.write(f'\n    {zone}_{metal},{slope:>10.6f}')
	
	print(f'[SAVED] {filename}')

# Save regressor to file
filename = 'NO.joblib'
dump(reg, filename)
print(f'[SAVED] {filename}')

def get_zone_multiplicity(n_atoms_zone, n_each_element):	
	'Return the multiplicity of the zone'
	return fac(n_atoms_zone) / np.prod([fac(n_each_element[idx_elem]) for idx_elem in range(n_metals)])

# Get number of unique set of features for each ensemble
n_config = int(np.prod([fac(n_atoms_zone + n_metals - 1) / (fac(n_atoms_zone) * fac(n_metals - 1)) for n_atoms_zone in n_atoms_zones]))

# Iterate through adsorption site ensembles
for ensemble, metal in zip(ensembles, metals):
	
	# Initiate container for features and multiplicities
	features = np.zeros((n_config, n_metals * n_zones), dtype=int)
	multiplicities = np.zeros(n_config, dtype=int)
	idx = 0
	
	# Iterate through all combinations of 1st zone
	for elems_1st in it.combinations_with_replacement(range(n_metals), n_atoms_zones[0]):
		
		# Count the number of each element in 1st zone
		count_1st = Counter(elems_1st)
		
		# Get multiplicity of the current configuration
		mult_1st = get_zone_multiplicity(n_atoms_zones[0], count_1st)
		
		# Iterate through all combinations of 2nd zone
		for elems_2nd in it.combinations_with_replacement(range(n_metals), n_atoms_zones[1]):
		
			# Count the number of each element in 2nd zone
			count_2nd = Counter(elems_2nd)
			
			# Get multiplicity of the current configuration
			mult_2nd = get_zone_multiplicity(n_atoms_zones[1], count_2nd)
			
			# Iterate through all combinations of 3rd zone
			for elems_3rd in it.combinations_with_replacement(range(n_metals), n_atoms_zones[2]):
		
				# Count the number of each element in 2nd zone
				count_3rd = Counter(elems_3rd)
				
				# Get multiplicity of the current configuration
				mult_3rd = get_zone_multiplicity(n_atoms_zones[2], count_3rd)
				
				# Append the element counts to the set of unique features
				features[idx] = [count[idx_metal] for count in [count_1st, count_2nd, count_3rd] for idx_metal in range(n_metals)]
				
				# Get multiplicity of the current set of features
				multiplicities[idx] = mult_1st * mult_2nd * mult_3rd

				# Increment configuration index
				idx += 1
				
	# Predict adsorption energy for features
	preds = reg.predict(ensemble, features)
	
	# Write predictions to file
	filename = f'NO_{metal}_all_sites.csv'
	out = np.concatenate((features, preds.reshape(-1, 1), multiplicities.reshape(-1, 1)), axis=1)
	fmt = '%d,' * (n_metals*n_zones) + '%.6f,' + '%d'
	header = ' '*(2*n_metals*n_zones-11) + 'features, prediction, multiplicity'
	np.savetxt(filename, out, fmt, header=header)
	print(f'[SAVED] {filename}')
