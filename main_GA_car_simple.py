import numpy as np
from classes import *
from filters import *
from GA_functions import *
from plotting_car import *
from helpers import seedRNG

'''
Holonomic Robot Simple Example

This is an extremely simple linear system where we have effectively a point robot with 
pose (x, y, theta), and because it is holonomic, it can move independently in each of the
three DOFs. Therefore, there are no nonlinear sinusoids that need to be linearized with
an EKF

We assume that the state is directly observable, but with a considerable amount of noise. 
So, this is like we have a noisy GPS estimate of the x,y position with a compass telling
us our heading angle

'''
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
duration = 20
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
k_selection = 5 # # Optional parameter, for tournament or truncation selection
L_interp_crossover = 0.5 # Optional parameter, for interpolation crossover
mutation_stdev = 0.25
regularization_scaling = 1
# Chromosomes
true_chromosome = A.flatten()
mu0_chromosome = np.eye(3).flatten() # TODO change this to make it not the actual value
sigma0_chromosome = 5*np.eye(len(mu0_chromosome)) # TODO: adjust the scaling on this


# Defining the genetic algorithm selection, crossover, and mutation methods
selection_fxn = rouletteWheelSelection
crossover_fxn = interpolationCrossover
mutation_fxn = gaussianMutation

# Seed the random number gen. for repeatable testing
seedRNG(0) 

# Sample the prior
x0 = np.random.multivariate_normal(mu0_state, sigma0_state)

# Build our models
time_model = TimeModel(dt, duration=duration)
dynamics_model = LinearDynamicsModel(A, B)
measurement_model = LinearMeasurementModel(C)
controls_model = ControlsModel(uHistory=np.vstack((vx_avg*abs(np.sin(time_model.times)), 
                                                   vy_avg*np.cos(time_model.times), 
                                                   phi_avg*np.sin(time_model.times))))
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
plot_generation(sim.xHistory, GA.muHistories_history, GA.evalHistory, gen=0) # Try adjusting the generation
plot_mahalanobis_convergence(GA.evalHistory)
plot_chromosome_convergence(GA.bestHistory, true_chromosome)


