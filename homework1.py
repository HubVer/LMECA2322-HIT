import numpy as np
import matplotlib.pyplot as plt
import scipy
import os



nu = 1.10555e-5 # kinematic viscosity used  (m^2/s)

u_velocities = np.zeros((3, 4, 4, 32768))

directories = sorted(os.listdir('Data'))[2:] # index to remove some unwanted files
for i, dir in enumerate(directories):
    files = sorted(os.listdir('Data/'+dir))[1:] # index to remove unwanted files

    for file in files:
        j, k = int(file[2]), int(file[4]) # get indices from file name
        u_velocities[i][j][k] = np.loadtxt('Data/'+dir+'/'+file)[:,i] # get u velocity



print(u_velocities)