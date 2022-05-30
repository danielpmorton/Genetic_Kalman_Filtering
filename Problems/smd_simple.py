import numpy as np

##############################################################################
# System Parameters ##########################################################
##############################################################################

dt = 0.1
k = 5
c = 1
m = 10
natural_freq = np.sqrt(k/m)
forcing_freq = 1.5*natural_freq

xdim = 2
ydim = 1
Q = 0.1 * np.eye(xdim) * dt
R = 0.1 * np.eye(ydim) 


mu0 = np.array([0,0])
sigma0 = 0.01 * np.eye(xdim)

# Other constants / useful values
duration = 50 # Length of the simulation
nTimesteps = np.floor(duration / dt).astype('int32')

# Do we need these??
eps = 1e-3 # For iEKF
maxIterations = 30 # For iEKF

##############################################################################
# Functions and Jacobians ####################################################
##############################################################################

def f(x,u):
    pos, vel = x
    force = u[0]
    d_pos = vel*dt
    d_vel = (1/m) * (force - c*vel - k*pos) * dt
    dx = np.array([d_pos, d_vel])
    return x + dx

def g(x):
    pos, vel = x
    return pos

def A(x, u):
    pos, vel = x
    return np.array([[      1,       dt], 
                     [-k*dt/m, 1-c*dt/m]])

def C(x):
    return np.array([[1, 0]])