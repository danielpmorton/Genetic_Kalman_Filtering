import numpy as np

##############################################################################
# System Parameters ##########################################################
##############################################################################

dt = 0.1
k = 5
c = 1
m_unknown = 10
natural_freq = np.sqrt(k/m_unknown)
forcing_freq = 1.5*natural_freq

xdim = 3 # Since we have position, velocity, and mass in the state now
ydim = 1 # Still just recording the position of the mass only
Q = 0.1 * np.eye(xdim) * dt
R = 0.1 * np.eye(ydim) 

# When simulating, there is no noise added to the unknown mass
Q_sim = 0.1 * dt * np.diag([1, 1, 0])
sigma0_sim = 0.01 * np.diag([1, 1, 0])
mu0_sim = np.array([0, 0, 10])

mu0 = np.array([0, 0, 10]) # Check trying initializing m past m
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

def f(x, u):
    pos, vel, m = x
    force = u[0]
    d_pos = vel*dt
    d_vel = (1/m) * (force - c*vel - k*pos) * dt
    d_mass = 0
    dx = np.array([d_pos, d_vel, d_mass])
    return x + dx

def g(x):
    pos, vel, m = x
    return pos

def A(x, u):
    pos, vel, m = x
    force = u[0]
    return np.array([[      1,       dt,                                     0], 
                     [-k*dt/m, 1-c*dt/m, (-1/(m**2)) * (force - c*vel - k*pos)], 
                     [      0,        0,                                     1]])

def C(x):
    return np.array([[1, 0, 0]])
