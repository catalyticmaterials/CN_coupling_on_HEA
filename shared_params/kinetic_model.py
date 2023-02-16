import numpy as np

# Define Boltzmann's constant
#kB = 1.380649e-4 / 1.602176634  # eV K-1 (exact)
kB=8.617333262 * 1e-5 #eV/K
kBT = kB*300 # eV
#Define Planck's constant
h = 4.135667696 * 1e-15 #eV*s


def urea_rate(DG_CO,DG_NO,eU,P_NO=1,Ea=None):
    E_CO = DG_CO - 0.4
    
    DG_N = (DG_NO-0.54)/0.65
    
    P_CO = 1
    P_H2O = 1
    
    #eU = -(DG_N - DG_NO)/2
    
    #Reaction constants
    K1 = np.exp(-DG_NO/(kBT))

    K2 = np.exp(-(DG_N-DG_NO+2*eU)/kBT)
    
    K3 = np.exp(-DG_CO/kBT)
    if Ea is None:
        DE = -2.09*(E_CO) - 2.88
        Ea = 0.46*DE + 1.49
    k4 = kBT/h * np.exp(-(Ea)/kBT)
    
    theta = 1/(1 + K1*P_NO + K3*P_CO + K1*K2*P_NO/P_H2O)
    
    #Reaction rate
    R_urea = k4*K1*K2*K3*P_NO*P_CO/P_H2O*theta**2
    
    return R_urea


def urea_conversion(DG_CO,DG_NO,eU,P_NO=1,Ea=None):
    E_NO = DG_NO - 0.71
    E_CO = DG_CO - 0.4
    
    DG_N = (DG_NO-0.54)/0.65
    DG_NOH = 0.77*DG_N + 1.01
    
    
    P_CO = 1
    P_H2O = 1
    
    #eU = -(DG_N - DG_NO)/2
    
    #Reaction constants
    K1 = np.exp(-DG_NO/(kBT))

    K2 = np.exp(-(DG_N-DG_NO+2*eU)/kBT)
    
    K3 = np.exp(-DG_CO/kBT)
    
    if Ea is None:
        DE = -2.09*(E_CO) - 2.88
        Ea = 0.46*DE + 1.49
    k4 = kBT/h * np.exp(-(Ea)/kBT)
    
    k6 = kBT/h * np.exp(-(DG_NOH-DG_NO+eU)/kBT)
    
    theta = 1/(1 + K1*P_NO + K3*P_CO + K1*K2*P_NO/P_H2O)
    
    #Reaction rates
    R_urea = k4*K1*K2*K3*P_NO*P_CO/P_H2O*theta**2

    R_NH3 = K1*k6*theta*P_NO
    
    urea_conversion = R_urea/(R_urea + R_NH3)
    return urea_conversion


def fractional_urea_rate(DG_CO,DG_NO,eU,P_NO=1,Ea=None):
    E_NO = DG_NO - 0.71
    E_CO = DG_CO - 0.4
    
    DG_N = (DG_NO-0.54)/0.65
    DG_NOH = 0.77*DG_N + 1.01
    
    
    P_CO = 1
    P_H2O = 1
    
    #eU = -(DG_N - DG_NO)/2
    
    #Reaction constants
    K1 = np.exp(-DG_NO/(kBT))

    K2 = np.exp(-(DG_N-DG_NO+2*eU)/kBT)
    
    K3 = np.exp(-DG_CO/kBT)
    
    if Ea is None:
        DE = -2.09*(E_CO) - 2.88
        Ea = 0.46*DE + 1.49
    k4 = kBT/h * np.exp(-(Ea)/kBT)
    
    k6 = kBT/h * np.exp(-(DG_NOH-DG_NO+eU)/kBT)
    
    theta = 1/(1 + K1*P_NO + K3*P_CO + K1*K2*P_NO/P_H2O)
    
    #Reaction rates
    R_urea = k4*K1*K2*K3*P_NO*P_CO/P_H2O*theta**2

    R_NH3 = K1*k6*theta*P_NO
    
    urea_conversion = R_urea/(R_urea + R_NH3)
    
    fractional_rate = R_urea*urea_conversion
    
    return fractional_rate


def activity(fcc_grid,top_grid,eU):
    CO_ids = np.array(np.nonzero(top_grid<0)).T
    ads_energy_pair = np.empty((0,2))

    #pad grids
    fcc_grid = np.pad(fcc_grid,pad_width=1,mode="wrap")
    top_grid = np.pad(top_grid,pad_width=1,mode="wrap")
    CO_ids+=1


    #Get all pairs of catalytic sites
    for (i,j) in CO_ids:
        if fcc_grid[i-1,j-1] < 0:
            E_CO = E_CO = top_grid[i,j]#CO_energies[i*100+j]
            E_NO = fcc_grid[i-1,j-1] #NO_energies[(i-1)*100+(j-1)]
            ads_energy_pair = np.vstack((ads_energy_pair,np.array([[E_CO,E_NO]])))
        if fcc_grid[i-1,j+1] < 0:
            E_CO = E_CO = top_grid[i,j]#CO_energies[i*100+j]
            E_NO = fcc_grid[i-1,j+1] #NO_energies[(i-1)*100+(j+1)]
            ads_energy_pair = np.vstack((ads_energy_pair,np.array([[E_CO,E_NO]])))
        if fcc_grid[i+1,j-1] < 0:
            E_CO = top_grid[i,j] #CO_energies[i*100+j]
            E_NO = fcc_grid[i+1,j-1] #NO_energies[(i+1)*100+(j-1)]
            ads_energy_pair = np.vstack((ads_energy_pair,np.array([[E_CO,E_NO]])))

    
    sum_activity = 0
    for (E_CO,E_NO) in ads_energy_pair:
        sum_activity += urea_conversion(E_CO, E_NO,eU)
    
    #Get activity as actitvity per surface atom
    activity = sum_activity/10000
    return activity

