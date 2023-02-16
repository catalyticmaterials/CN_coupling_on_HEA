import numpy as np
from ase.db import connect
from ase.data import covalent_radii,vdw_radii
import matplotlib.pyplot as plt

not_ontop=0

z_vec = np.array([0,0,1])
angles = np.empty(0)
not_ontop_angles = np.empty(0)
ontop_angles = np.empty((0,2))

metals = ['Ag','Au', 'Cu', 'Pd','Pt']

with connect("NO_out_sorted.db") as db:
    for row in db.select():
        atoms=db.get_atoms(row.id)
        site_idx=row.site_idx
        
        # Repeat atoms object
        atoms_3x3 = atoms.repeat((3,3,1))
		
		# Get chemical symbols of atoms
        symbols = np.asarray(atoms_3x3.get_chemical_symbols())
		
		# Get index of central hydrogen atom
        idx_N = np.nonzero(symbols == 'N')[0][4]
		
		# Get position of carbon atom
        pos_N = atoms_3x3.positions[idx_N]
		
		# Get indices of the 1st layer atoms
        ids_1st = np.array([atom.index for atom in atoms_3x3 if atom.tag == 1])
		
		# Get squared distances to the 1st layer atoms from the hydrogen atom
        dists_sq = np.sum((atoms_3x3.positions[ids_1st] - pos_N)**2, axis=1)
		
        # Get the three shortest distances
        dist_3 = np.sort(np.sqrt(dists_sq))[:3]  
        
        # Get one atom from 4th and 5th layer 
        id_4th = np.array([atom.index for atom in atoms_3x3 if atom.tag == 4])[0]
        id_5th = np.array([atom.index for atom in atoms_3x3 if atom.tag == 5])[7]
        
        # Get distance in z
        z_dist = atoms_3x3.positions[id_4th][2] - atoms_3x3.positions[id_5th][2]
        
        # Set average radius of atoms
        r = z_dist/2
        # Get radius of N
        r_N = covalent_radii[7]
        
        #Set reference distance
        r_ref = r + r_N
        
        #Get number of distances below reference +20%
        n = np.sum(dist_3 < r_ref*1.2)
        
        
        # Get index of central Oxygen atom
        idx_O = np.nonzero(symbols == 'O')[0][4]
		
		# Get position of carbon atom
        pos_O = atoms_3x3.positions[idx_O]
        
        NO_vec = pos_O - pos_N
        
        angle = np.arccos(np.dot(z_vec,NO_vec)/(np.linalg.norm(NO_vec)*np.linalg.norm(z_vec)))
        angle*=180/np.pi
        
        
        if n>1:
            not_ontop+=1
            #print(row.id,row.site_idx)
            not_ontop_angles = np.append(not_ontop_angles,angle)
        else:
            
            idx_site = ids_1st[np.argmin(dists_sq)]
            
            metal = symbols[idx_site]
            append = np.array([[angle,metals.index(metal)]])
            ontop_angles = np.vstack((ontop_angles,append))


#print(ontop,bridge,hollow)
#print(sum([ontop,bridge,hollow]))
#print("not ontop:",not_ontop)
plt.figure(dpi=400)
plt.hist(ontop_angles[:,0],bins=20,histtype="step",color="tab:blue",label="Ontop sites")
plt.hist(not_ontop_angles,bins=20,histtype="step",color="tab:orange",label= "Not ontop sites")
plt.legend(loc=2)
plt.xlabel("N-O angle (degrees)")
plt.ylabel("Counts")


plt.figure(dpi=400)
#mask_Au = ontop_angles[:,1]==0
#mask_Ag = ontop_angles[:,1]==1
#mask_Cu = ontop_angles[:,1]==2
#mask_Pd = ontop_angles[:,1]==3
#mask_Pt = ontop_angles[:,1]==4
mean_angles=[]
for i,metal in enumerate(metals):
    mask = ontop_angles[:,1]==i
    angles = ontop_angles[:,0][mask]
    plt.hist(angles,bins=20,histtype="step",label=metal)
    mean_angles.append(np.mean(angles))
    print(metal,": Mean angle: ", np.mean(angles))
plt.legend()
plt.xlabel("N-O angle (degrees)")
plt.ylabel("Counts")
Ag,Au,Cu,Pd,Pt = mean_angles
s=f"Mean angles (deg): \nAg={Ag:.2f}\nAu={Au:.2f}\nCu={Cu:.2f}\nPd={Pd:.2f}\nPt={Pt:.2f}"
plt.text(x=9,y=12.5,s=s)

print("\nsingle element slabs")
#single element slabs
with connect("single_element_NO_angled_out.db") as db:
    for row in db.select():
        metal = row.metal
        atoms=db.get_atoms(row.id)
        
        # Repeat atoms object
        atoms_3x3 = atoms.repeat((3,3,1))
		
		# Get chemical symbols of atoms
        symbols = np.asarray(atoms_3x3.get_chemical_symbols())
		
		# Get index of central hydrogen atom
        idx_N = np.nonzero(symbols == 'N')[0][4]
		
		# Get position of carbon atom
        pos_N = atoms_3x3.positions[idx_N]
		
		# Get indices of the 1st layer atoms
        ids_1st = np.array([atom.index for atom in atoms_3x3 if atom.tag == 1])
		
		# Get squared distances to the 1st layer atoms from the hydrogen atom
        dists_sq = np.sum((atoms_3x3.positions[ids_1st] - pos_N)**2, axis=1)
		
        # Get the three shortest distances
        dist_3 = np.sort(np.sqrt(dists_sq))[:3]  
        
        # Get index of central Oxygen atom
        idx_O = np.nonzero(symbols == 'O')[0][4]
		
		# Get position of carbon atom
        pos_O = atoms_3x3.positions[idx_O]
        
        NO_vec = pos_O - pos_N
        print(np.linalg.norm(NO_vec))
        angle = np.arccos(np.dot(z_vec,NO_vec)/(np.linalg.norm(NO_vec)*np.linalg.norm(z_vec)))
        angle*=180/np.pi
        
        print(metal,": angle: ",angle)
