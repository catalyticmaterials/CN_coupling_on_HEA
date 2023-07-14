from sklearn.linear_model import LinearRegression
from copy import deepcopy
import numpy as np

class LinearRegressor():
	
	def __init__(self,  n_ensembles, n_atoms_zones, n_metals):
		self.n_ensembles = n_ensembles
		self.n_atoms_zones = n_atoms_zones
		self.n_metals = n_metals
	
	def fit(self, X, y):
		'Fit linear regressor and adjust linear parameters'
		
		# Specify and fit linear regressor
		reg = LinearRegression(fit_intercept=False).fit(X, y)
		
		# Get ensemble parameters
		ens_slopes = reg.coef_[:self.n_ensembles]
		
		# Get remaining parameters
		slopes = reg.coef_[self.n_ensembles:]

		# Iterate through zones
		for zone_idx, n_atoms_zone in enumerate(self.n_atoms_zones):
	
			mean = np.mean(slopes[zone_idx*self.n_metals : (zone_idx+1)*self.n_metals])
			ens_slopes += n_atoms_zone * mean
			slopes[zone_idx*self.n_metals : (zone_idx+1)*self.n_metals] -= mean
		
		self.params = np.concatenate((ens_slopes, slopes))
		
	def predict(self, X):
		'Predict on X'
		return self.params @ X.T


class MultiLinearRegressor():
	'Class for doing multilinear regression'
	
	def __init__(self, n_atoms_zones):
		self.n_atoms_zones = n_atoms_zones
		self.slopes = {}
		self.intercepts = {}
		
	def fit(self, ensemble, X, y):
		'Train regressor on X and y for the specified ensemble'
		reg = LinearRegression(fit_intercept=True).fit(X, y)
		
		## Adjust coefficients so that the pure metal itself has zero-parameters
		
		# ..get trained coefficients
		slopes_orig = reg.coef_
		slopes = deepcopy(slopes_orig)
		
		# ..get trained intercept
		intercept = reg.intercept_
		
		# ..get number of metals
		n_metals = len(ensemble)
		
		# ..get index of metal from the ensemble
		idx_ens_metal = ensemble.index(1)
		
		# ..subtract element parameters from the parameters
		# of the other elements in each zone
		for idx_zone, n_metals_zone in enumerate(self.n_atoms_zones):
			
			# Iterate through metal indices
			for idx_metal in range(n_metals):
				
				# Subtract from the slopes
				slopes[idx_zone * n_metals + idx_metal] -= slopes_orig[idx_zone * n_metals + idx_ens_metal]
			
			# Add to the intercept
			intercept += n_metals_zone * slopes_orig[idx_zone * n_metals + idx_ens_metal]
		
		# Update slopes and intercept for this ensemble
		self.slopes[ensemble] = slopes
		self.intercepts[ensemble] = intercept
		
	def predict(self, ensemble, X):
		'Predict on X for the specified ensemble'
		return self.intercepts[ensemble] + self.slopes[ensemble] @ X.T
	
	def get_parameters(self):
		'Return linear regression parameters'
		return self.intercepts, self.slopes
