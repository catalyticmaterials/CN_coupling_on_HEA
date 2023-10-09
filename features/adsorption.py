import numpy as np
import matplotlib.pyplot as plt

CO = np.loadtxt('CO.csv',delimiter=',',skiprows=1,usecols=(20))

NO_top = np.loadtxt('NO.csv',delimiter=',',skiprows=1,usecols=(20))

NO_fcc = np.loadtxt('NO_fcc.csv',delimiter=',',skiprows=1,usecols=(45))

H = np.loadtxt('H.csv',delimiter=',',skiprows=1,usecols=(45))


fig, ax  = plt.subplots()

ax.hist(NO_top,histtype='step',bins=20)
ax.hist(NO_fcc,histtype='step',bins=20)

plt.show()