import numpy as np
from classes import *
from filters import *
from Problems.smd_simple import *
from helpers import seedRNG, plot_SMD

seedRNG(0)

# Sample the priors
x0 = np.random.multivariate_normal(mu0, sigma0) # Sample x0 from the prior

# Build our models
time_model = TimeModel(dt, duration=duration)
dynamics_model = DynamicsModel(f, A, xdim)
measurement_model = MeasurementModel(g, C, ydim)
force = lambda t : np.sin(forcing_freq*t)
controls_model = ControlsModel(fxn=force, times=time_model.times)
noise_model = NoiseModel(Q, R)
P = Problem(dynamics_model, measurement_model, controls_model, noise_model, time_model)
sim = SimulatedResult(x0, P)
# Evaluate the filter
storage = FilterStorage(mu0, sigma0, time_model.nTimesteps)
filtered = FilterResult(EKF, P, sim.yHistory, storage)
# filtered = FilterResult(iEKF, P, sim.yHistory, storage, iEKF_maxIter=20, iEKF_eps=1e-3)
# filtered = FilterResult(UKF, P, sim.yHistory, storage)

plot_SMD(time_model.times, sim.xHistory, filtered.muHistory, filtered.sigmaHistory, 
         f"{filtered.filter.__name__} Spring Mass Damper", ["Position", "Velocity"])