from matplotlib.colors import to_rgb
import itertools as it
import numpy as np

# Specify colors of metals
metal_colors = dict(Ag='silver',
					Au='gold',
					Cu='darkorange',
					Pd='royalblue',
					Pt='green')

# Update colors for pairs and triples of sites
for n_elems in [2, 3]:

	for elem_comb in it.combinations_with_replacement(metal_colors.keys(), n_elems):
		
		colors_elems = np.asarray([to_rgb(metal_colors[elem]) for elem in elem_comb])
		color_comb = np.mean(colors_elems, axis=0)
		
		# Gert alphabetically sorted string of elements
		comb_str = ''.join(sorted(elem_comb))
		
		# Add color of combination to the dictionary of colors
		metal_colors[comb_str] = tuple(color_comb)
