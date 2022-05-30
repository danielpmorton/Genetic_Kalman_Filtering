import numpy as np

'''
Holonomic Robot: Setting up the parameters for the system
'''

##############################################################################
# System Parameters ##########################################################

# Dimensions
xdim = 3
ydim = 3
chromdim = xdim**2
# For controls
vx_avg = 1
vy_avg = 2
phi_avg = .1
# Timing
duration = 10
dt = 0.1
# Initial estimate for the filters
mu0_state = np.zeros(3)
sigma0_state = 0.01 * np.ones((3,3))
# Matrices
A = np.eye(3)
B = dt*np.eye(3)
C = np.eye(3)
Q = 0.1 * np.eye(xdim) * dt
R = 0.1 * np.eye(ydim)
# Genetic Algos
pop_size = 10 # Number of KFs we have running
k_max = 20 # Maximum number of iterations in the GA
mutation_stdev = 0.25
# Chromosomes
true_chromosome = A.flatten()
mu0_chromosome = np.eye(3).flatten() # TODO change this to make it not the actual value
sigma0_chromosome = 5*np.eye(len(mu0_chromosome)) # TODO: adjust the scaling on this