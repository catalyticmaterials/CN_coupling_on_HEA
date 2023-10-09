from ase.db import connect
from ase.io.trajectory import Trajectory
from ase.io.ulm import InvalidULMFileError
import os

# Set calculator parameters
calc_params = {'mode': {'name': 'pw', 'ecut': 400.0, 'gammacentered': False},
			   'xc': 'BEEF-vdW',
			   'kpts': [4, 4, 1],
			   'eigensolver': 'dav',
			   'parallel': {'augment_grids': True, 'sl_auto': True}}

# Iterate through files in the current folder
for db_name in os.listdir():

	# If the file name ends with '.db' but not '_out.db', then proceed
	if db_name.endswith('.db') and not db_name.endswith('_out.db'):
		
		# Get base name
		db_root = db_name.split('.')[0]
		
		# Set name of output database
		db_out_name = db_root + '_out.db'
		print('db_out_name:', db_out_name)

		# Load input and output databases
		with connect(db_name) as db, connect(db_out_name) as db_out:

			# Iterate through database entries
			for row in db.select():
				
				# Append row index to base filename
				file_root = f'{db_root}_{str(row.id).zfill(3)}'
				
				try:
				
					# If an empty error file exists for this simulation,
					# then this simulation has probably run successfully already
					# and will be skipped here
					if os.stat(f'calc/{file_root}.err').st_size == 0:
						continue
				
				# If the error file does not exist then go ahead
				except FileNotFoundError:
					pass
				
				# If the sample is already in the output database and has an energy,
				# then continue to the next row
				try:
					
					db_out.get('energy', idx=row.id)
					continue

				# Else, if the sample is not in the database,
				# then proceed with making a file for it
				except KeyError:
					db_name_ = db_name
				
					# Reserve a row in the database with the same row index
					idx = db_out.reserve(idx=row.id)
					
				# Get keyword-value pairs of the current row
				# to be passed on to the output database
				row_kwargs = {key: row[key] for key in row._keys}				
				
				# If a trajectory file has already been written,
				# then proceed from that
				try:
					traj = Trajectory(f'calc/{file_root}.traj')
					read_traj = True
				
				except (FileNotFoundError, InvalidULMFileError):
					read_traj = False

				# Write python script
				with open(f'relax/{file_root}.py', 'w') as file_:
			
					file_.writelines([
					"from ase.db import connect\n",
					"from gpaw import GPAW, PW\n",
					"from gpaw import MixerDif\n",
					"from time import sleep\n",
					"from sqlite3 import OperationalError\n"
					])

					if read_traj:
						file_.writelines([
						"from ase.io.trajectory import Trajectory\n",
						"\n",
						f"traj = Trajectory('calc/{file_root}.traj')\n"
						f"atoms = traj[-1]\n"
						])
					
					else:
						file_.writelines([
						"\n",
						f"with connect('{db_name_}') as db:\n",
						f"\tatoms = db.get_atoms({row.id})\n"
						])
					
					file_.writelines([
					"\n",
					f"calc = GPAW(mode=PW(800),kpts=(20, 20, 20),xc='BEEF-vdW',mixer=MixerDif(beta=0.05, nmaxold=5, weight=50.0),txt='calc/{file_root}.txt')\n",
					"atoms.set_calculator(calc)\n",
					"\n",
					"atoms.get_potential_energy()",
					"\n",
					"while True:\n",
					"\ttry:\n",
					f"\t\twith connect('{db_root}_out.db') as db:\n",
					f"\t\t\tdb.write(atoms, id={row.id}, **{row_kwargs})\n",
					"\t\tbreak\n",
					"\texcept OperationalError:\n",
					"\t\tsleep(3)"
					])
		
				# Write slurm submission script
				with open(f'submit/{file_root}.sl', 'w') as file_:
				
					file_.writelines([
					"#!/bin/bash\n",
					f"#SBATCH --job-name={file_root}\n",
					"#SBATCH --partition=katla_short\n",
					f"#SBATCH --output=calc/{file_root}.log\n",
					f"#SBATCH --error=calc/{file_root}.err\n",
					"#SBATCH --nodes=1\n",
					"#SBATCH --ntasks=16\n",
					"#SBATCH --ntasks-per-core=2\n",
					"#SBATCH --mem-per-cpu=2G\n",
					"module purge\n",
					'. "/groups/kemi/jack/miniconda3/etc/profile.d/conda.sh"\n',
					"conda activate gpaw22\n",
					"export OMP_NUM_THREADS=1\n",
					"export OMPI_MCA_pml=\"^ucx\"\n",
					"export OMPI_MCA_osc=\"^ucx\"\n",
					f"mpirun --mca btl_openib_rroce_enable 1 gpaw python relax/{file_root}.py"
					])
