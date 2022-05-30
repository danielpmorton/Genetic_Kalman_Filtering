import numpy as np
from classes import *
from filters import *
from Problems.car_slam import *
from helpers import seedRNG, plotSLAM # Update this in the future - import more stuff/all?

# Testing with code from HW6/HW7

# Using a random seed to make the plots and trajectories more consistent across tests
# Arbitrary. SLAM: 0,11,13 ok, 12 good, 16 very good
seedRNG(16)

# Sample the priors
x0 = np.random.multivariate_normal(mu0, sigma0) # Sample x0 from the prior
x0_sim = np.random.multivariate_normal(mu0, sigma0_sim)

# Build our models
time_model = TimeModel(dt, duration=20)
dynamics_model = NonlinearDynamicsModel(f, A, xdim)
measurement_model = MeasurementModel(g, C, ydim)
controls_model = ControlsModel(uHistory=np.vstack((vt*np.ones(time_model.nTimesteps), np.sin(time_model.times))))
noise_model = NoiseModel(Q, R)
P = Problem(dynamics_model, measurement_model, controls_model, noise_model, time_model)

# Simulation will use modified values because we know the map exactly with SLAM
# This is going to be problem-dependent
noise_model_sim = NoiseModel(Q_sim, R)
P_sim = Problem(dynamics_model, measurement_model, controls_model, noise_model_sim, time_model)
sim = SimulatedResult(x0_sim, P_sim)

# Evaluate the filter
storage = FilterStorage(mu0, sigma0, time_model.nTimesteps)
filtered = FilterResult(EKF, P, sim.yHistory, storage)
# filtered = FilterResult(iEKF, P, sim.yHistory, storage, iEKF_maxIter=20, iEKF_eps=1e-3)
# filtered = FilterResult(UKF, P, sim.yHistory, storage)

# from plotting_car import plot
# plot(time_model.times, sim.xHistory[:3], filtered.muHistory[:3], filtered.sigmaHistory[:3,:3], "Test")

plotSLAM(sim.xHistory, filtered.muHistory, f"{filtered.filter.__name__} SLAM")