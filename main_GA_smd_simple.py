import numpy as np
from classes import *
from filters import *
from GA_functions import *
from plotting_smd import *
from helpers import seedRNG

'''
Spring Mass Damper Simple Example

Another simple linear system

We assume that the position is observable

TODO Finish description

'''
# System Parameters ##########################################################
# Dimensions
xdim = 2
ydim = 1
chromdim = xdim**2
# Timing
duration = 10
dt = 0.1
# Initial estimate for the filters
mu0_state = np.zeros(xdim)
sigma0_state = 0.01 * np.ones((xdim,xdim))
# True system parameters (only for simulation)
k = 5
c = 1
m = 10
natural_freq = np.sqrt(k/m)
# For controls
force_mag = 2
forcing_freq = 1.5*natural_freq 
# Matrices
A = np.array([[      1,       dt], 
              [-k*dt/m, 1-c*dt/m]]) # Only for simulation
B = np.array([0, dt/m]).reshape(-1,1)
C = np.array([[1,0]])
Q = 0.1 * np.eye(xdim) * dt
R = 0.1 * np.eye(ydim)
# Genetic Algos
pop_size = 10 # Number of KFs we have running
k_max = 20 # Maximum number of iterations in the GA
k_selection = 5 # Optional parameter, for tournament or truncation selection
L_interp_crossover = 0.5 # Optional parameter, for interpolation crossover
mutation_stdev = 0.25
regularization_scaling = 0.25
# Chromosomes
true_chromosome = A.flatten()
mu0_chromosome = A.flatten() # TODO change this to make it not the actual value
sigma0_chromosome = 5*np.eye(len(mu0_chromosome)) # TODO: adjust the scaling on this


# Defining the genetic algorithm selection, crossover, and mutation methods
selection_fxn =  rouletteWheelSelection
crossover_fxn = uniformCrossover # interpolationCrossover
mutation_fxn = gaussianMutation

# Seed the random number gen. for repeatable testing
seedRNG(0) 

# Sample the prior
x0 = np.random.multivariate_normal(mu0_state, sigma0_state)

# Build our models
time_model = TimeModel(dt, duration=duration)
dynamics_model = LinearDynamicsModel(A, B)
measurement_model = LinearMeasurementModel(C)
controls_model = ControlsModel(uHistory=np.array([force_mag*np.sin(forcing_freq * time_model.times)]))
noise_model = NoiseModel(Q, R)
P = Problem(dynamics_model, measurement_model, controls_model, noise_model, time_model)
sim = SimulatedResult(x0, P)

# Running the Genetic Algorithm
GAinit = GAInitialization(mu0_state, sigma0_state, mu0_chromosome, sigma0_chromosome, 
                          time_model.nTimesteps, pop_size, selection_fxn, crossover_fxn, mutation_fxn, k_max, 
                          k_selection, L_interp_crossover, mutation_stdev, regularization_scaling)
GA = GAResult(KF, P, sim.yHistory, GAinit)

# Print the true A matrix for comparison
print(f"True A matrix: {A}")

# Running the Kalman filter with known A for comparison
storage = FilterStorage(mu0_state, sigma0_state, time_model.nTimesteps)
filtered = FilterResult(KF, P, sim.yHistory, storage)
plot_optimal_filter_vs_truth(time_model.times, sim.xHistory, filtered.muHistory, filtered.sigmaHistory)


# Plotting
last_gen_best_ind = np.argmin(GA.evalHistory[-1])
plot_best_filter_vs_truth(time_model.times, sim.xHistory, GA.muHistories[last_gen_best_ind], GA.sigmaHistories[last_gen_best_ind])
plot_generation(time_model.times, sim.xHistory, GA.muHistories_history, GA.evalHistory, gen=0) # Try adjusting the generation
plot_mahalanobis_convergence(GA.evalHistory)
plot_chromosome_convergence(GA.bestHistory, true_chromosome)

