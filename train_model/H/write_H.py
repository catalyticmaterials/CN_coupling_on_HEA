import numpy as np
import itertools as it
from collections import Counter
from math import factorial as fac
import sys
sys.path.append('../..')
from scripts import metals, n_metals
from scripts.regressor import LinearRegressor
from joblib import dump

# Get ensembles
n_atoms_ensemble = 3
ensembles = list(it.combinations_with_replacement(metals, n_atoms_ensemble))
ensembles_as_numbers = list(it.combinations_with_replacement(range(n_metals), n_atoms_ensemble))

# Get number of ensembles
n_ensembles = len(ensembles)

# Specify zone sizes
n_atoms_zones = [3, 3]
n_zones = len(n_atoms_zones)

# Read features from file
filename = '../../features/H.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)
features = data[:, :-4]
energies = data[:, -4]



# Define regressor
reg = LinearRegressor(n_ensembles, n_atoms_zones, n_metals)

# Train regressor
reg.fit(features, energies)

# Save regressor to file
filename = 'H.joblib'
dump(reg, filename)
print(f'[SAVED] {filename}')

# Get ensemble parameters
ens_slopes = reg.params[:n_ensembles]

# Get remaining parameters
slopes = reg.params[n_ensembles:]

# Save linear parameters to file
filename = f'H_linear_parameters.csv'
with open(filename, 'w') as file_:
	
	# Write ensemble parameters to file
	for ensemble, slope in zip(ensembles, ens_slopes):
		ens_str = ''.join(ensemble)
		file_.write(f'{ens_str},{slope:>10.6f}\n')
	
	# Write remaining linear parameters to file
	for zone_idx, zone in enumerate(['1a', '2a']):
		
		for slope, metal in zip(slopes[zone_idx*n_metals : (zone_idx+1)*n_metals], metals):
			file_.write(f' {zone}_{metal},{slope:>10.6f}\n')
	
	print(f'[SAVED] {filename}')

def get_zone_multiplicity(n_atoms_zone, n_each_element):	
	'Return the multiplicity of the zone'
	return fac(n_atoms_zone) / np.prod([fac(n_each_element[idx_elem]) for idx_elem in range(n_metals)])

# Get number of unique set of features for each ensemble
n_config = int(np.prod([fac(n_atoms_zone + n_metals - 1) / (fac(n_atoms_zone) * fac(n_metals - 1)) for n_atoms_zone in n_atoms_zones]))

# Initiate container for features and multiplicities
features = np.zeros((n_ensembles * n_config, n_ensembles + n_metals * n_zones), dtype=int)
multiplicities = np.zeros(n_ensembles * n_config, dtype=int)
idx = 0

# Iterate through adsorption site ensembles
for ens_idx, (ensemble, ensemble_as_numbers) in enumerate(zip(ensembles, ensembles_as_numbers)):
	
	# Count the number of each element in the ensemble
	count_ens = Counter(ensemble_as_numbers)
	
	# Get multiplicity of the current ensemble
	mult_ens = get_zone_multiplicity(n_atoms_ensemble, count_ens)
	
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
			
			# One hot encode the current ensemble in the features
			features[idx, ens_idx] = 1
			
			# Append the element counts to the set of unique features
			features[idx, n_ensembles:] = [count[idx_metal] for count in [count_1st, count_2nd] for idx_metal in range(n_metals)]
			
			# Get multiplicity of the current set of features
			multiplicities[idx] = mult_ens * mult_1st * mult_2nd

			# Increment configuration index
			idx += 1
				
# Predict adsorption energy for features
preds = reg.predict(features)

# Write predictions to file
filename = 'H_all_sites.csv'
out = np.concatenate((features, preds.reshape(-1, 1), multiplicities.reshape(-1, 1)), axis=1)
fmt = '%d,' * (n_ensembles + n_metals*n_zones) + '%.6f,' + '%d'
header = ' '*(2*(n_ensembles + n_metals*n_zones)-11) + 'features, prediction, multiplicity'
np.savetxt(filename, out, fmt, header=header)
print(f'[SAVED] {filename}')
