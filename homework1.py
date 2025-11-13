import numpy as np
import matplotlib.pyplot as plt
import scipy


nu = 1.10555e-5 # kinematic viscosity used  (m^2/s)

with open('Data/pencils_x/x_0_0.txt') as f:
    data = f.readlines()

print(data)