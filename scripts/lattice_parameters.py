import json
from pathlib import Path

# Get path to lattice parameters relative to the parent file (the executed file)
# This is to make relative import work
path = Path(__file__).parent / '../0_lattice_parameters/lattice_parameters.json'

# Read fcc lattice parameters from file
with path.open('r') as file_:
	lattice_parameters = json.load(file_)
