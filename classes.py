import numpy as np
import scipy.linalg
import copy
from math import isnan

class SigmaPoints:
    def __init__(self, mu, sigma, L=2):
        self.mu = mu
        self.sigma = sigma
        self.L = L # lambda
        self.points, self.weights = self.getPointsAndWeights()
        
    def getPointsAndWeights(self):
        xdim = len(self.mu)
        points = [] # A list of np arrays
        weights = [] # A list of values
        # Handle the central point
        center = self.mu
        centerweight = self.L / (self.L + xdim)
        points.append(center)
        weights.append(centerweight)
        # Handle the other points around the center
        for i in range(xdim):
            offset = scipy.linalg.sqrtm((self.L + xdim)*self.sigma)[:,i]
            point1 = self.mu + offset
            point2 = self.mu - offset
            weight = 1/(2 * (self.L + xdim))
            points.extend((point1, point2))
            weights.extend((weight, weight))
        return points, weights

class ParticleSet:
    def __init__(self, mu0, sigma0, numParticles):
        self.particles = np.random.multivariate_normal(mu0, sigma0, size=numParticles).T # With the transpose, shape = (xdim,numParticles)
        self.weights = 1/numParticles * np.ones(numParticles) # size (numParticles,)

# Combines the dynamics, measurement, controls, noise, and time into a single Problem model
class Problem:
    def __init__(self, dynamics_model, measurement_model, controls_model, noise_model, time_model):
        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model
        self.controls_model = controls_model
        self.noise_model = noise_model
        self.time_model = time_model

class NonlinearMeasurementModel:
    def __init__(self, g, C, ydim):
        self.g = g 
        self.C = C 
        self.ydim = ydim

    def g(self, x):
        return self.g(x)

    def C(self, x):
        return self.C(x)

class LinearMeasurementModel:
    def __init__(self, C):
        self.C = C 
        self.ydim = C.shape[0]

    def g(self, x):
        return self.C @ x

class NonlinearDynamicsModel:
    def __init__(self, f, A, xdim):
        self.f = f # Callable. f(x, u)
        self.A = A # Callable. A(x, u)
        self.xdim = xdim
    
    def f(self, x, u):
        return self.f(x, u)
    
    def A(self, x, u):
        return self.A(x, u)

class LinearDynamicsModel:
    def __init__(self, A, B):
        self.A = A # Matrix
        self.B = B
        self.xdim = A.shape[0]
    
    def f(self, x, u):
        return self.A @ x + self.B @ u

class NoiseModel:
    def __init__(self, Q, R):
        self.Q = Q
        self.R = R

class ControlsModel:
    # Can initialize this with either the full uHistory array
    # Or, a function applied over an array of times
    def __init__(self, uHistory=None, fxn=None, times=None):
        self.fxn = fxn
        self.times = times

        if uHistory is not None:
            self.uHistory = uHistory
        elif fxn is not None and times is not None:
            dim = 1 if np.isscalar(fxn(times[0])) else len(fxn(times[0]))
            self.uHistory = np.zeros((dim, len(times)))
            for i,t in enumerate(times):
                self.uHistory[:,i] = self.fxn(t)
        else:
            raise Exception("Must provide more inputs to the controls model")

class TimeModel:
    # Must provide dt and one of the three optional inputs
    def __init__(self, dt, nTimesteps=None, times=None, duration=None):
        self.dt = dt
        if duration != None:
            self.duration = duration
            self.nTimesteps = np.floor(self.duration / self.dt).astype('int32')
            self.times = self.dt * np.arange(self.nTimesteps)
        elif nTimesteps != None:
            self.nTimesteps = nTimesteps
            self.duration = self.dt * self.nTimesteps
            self.times = self.dt * np.arange(self.nTimesteps)
        elif times != None:
            self.times = times
            self.nTimesteps = len(times)
            self.duration = self.dt * self.nTimesteps
        else:
            raise Exception("Must provide more info to the time model!")


class FilterStorage:
    def __init__(self, mu0, sigma0, nTimesteps):
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.nTimesteps = nTimesteps
        self.muHistory, self.sigmaHistory = self.initializeStorage(mu0, sigma0, nTimesteps)
    
    def initializeStorage(self, mu0, sigma0, nTimesteps):
        dim = len(mu0)
        muHistory = np.zeros((dim, nTimesteps))
        sigmaHistory = np.zeros((dim, dim, nTimesteps))
        muHistory[:,0] = mu0
        sigmaHistory[:,:,0] = sigma0
        return muHistory, sigmaHistory

class PFStorage(FilterStorage):
    def __init__(self, mu0, sigma0, nTimesteps, numParticles):
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.numParticles = numParticles
        self.X = self.initParticleSet(mu0, sigma0, numParticles)
        self.muHistory, self.sigmaHistory = self.initializeStorage(mu0, sigma0, nTimesteps)

    def initParticleSet(mu0, sigma0, numParticles):
        particles = np.random.multivariate_normal(mu0, sigma0, size=numParticles).T # With the transpose, shape = (state dim,1000)
        weights = 1/numParticles * np.ones(numParticles)
        X = ParticleSet(particles, weights)
        return X

class SimulatedResult:
    def __init__(self, x0, problem):
        self.x0 = x0
        self.uHistory = problem.controls_model.uHistory
        self.dt = problem.time_model.dt
        self.nTimesteps = problem.time_model.nTimesteps
        self.times = problem.time_model.times
        self.R = problem.noise_model.R
        self.Q = problem.noise_model.Q
        self.xdim = len(self.x0)
        self.ydim = self.R.shape[0]
        self.f = problem.dynamics_model.f
        self.g = problem.measurement_model.g
        self.xHistory, self.yHistory = self.simulate()

    def simulate(self):
        xHistory = np.zeros((self.xdim, self.nTimesteps))
        xHistory[:,0] = self.x0
        yHistory = np.zeros((self.ydim, self.nTimesteps))

        for i in range(1, self.nTimesteps):
            # Retrieve
            xprev = xHistory[:,i-1]
            u = self.uHistory[:,i]
            # Calculate noise
            v = np.random.multivariate_normal(np.zeros(self.ydim), self.R)
            w = np.random.multivariate_normal(np.zeros(self.xdim), self.Q)
            # Calculate state and measurement
            xnew = self.f(xprev, u) + w
            y = self.g(xnew) + v
            # Store data
            xHistory[:,i] = xnew
            yHistory[:,i] = y
        
        print("Simulation complete")
        return xHistory, yHistory

class FilterResult:
    def __init__(self, filter, problem, measurement_history, filter_storage, UKF_lambda=2, iEKF_maxIter=None, iEKF_eps=None, PF_numParticles=None, PF_resample=True):
        self.Q = problem.noise_model.Q
        self.R = problem.noise_model.R
        self.f = problem.dynamics_model.f
        self.A = problem.dynamics_model.A
        self.g = problem.measurement_model.g
        self.C = problem.measurement_model.C
        self.uHistory = problem.controls_model.uHistory
        self.yHistory = measurement_history
        self.filter = filter
        self.muHistory = filter_storage.muHistory # Initial value
        self.sigmaHistory = filter_storage.sigmaHistory # Initial value
        self.nTimesteps = problem.time_model.nTimesteps
        self.UKF_lambda = UKF_lambda # Default value of 2 since this is standard
        self.iEKF_maxIter = iEKF_maxIter
        self.iEKF_eps = iEKF_eps
        self.PF_numParticles = PF_numParticles
        self.PF_resample = PF_resample
        self.runFilter()

    def runFilter(self):
        # Initialize mu and sigma to the mu0, sigma0 values stored
        mu = self.muHistory[:,0]
        sigma = self.sigmaHistory[:,:,0]
        # If we have a PF, need to also initialize a particle set
        if self.filter.__name__.lower() == "pf":
            X = ParticleSet(self.mu, self.sigma, self.PF_numParticles)
        # Filter for all timesteps
        for i in range(1, self.nTimesteps):
            # Retrieve from simulation data
            y = self.yHistory[:,i]
            u = self.uHistory[:,i]
            # Filter - TODO: need to check if the filter name thing works
            if self.filter.__name__.lower() == "kf":
                mu, sigma = self.filter(mu, sigma, u, y, self.Q, self.R, self.f, self.g, self.A, self.C)
            elif self.filter.__name__.lower() == "ekf":
                mu, sigma = self.filter(mu, sigma, u, y, self.Q, self.R, self.f, self.g, self.A, self.C)
            elif self.filter.__name__.lower() == "ukf":
                mu, sigma = self.filter(mu, sigma, u, y, self.Q, self.R, self.f, self.g, self.UKF_lambda)
            elif self.filter.__name__.lower() == "iekf":
                mu, sigma = self.filter(mu, sigma, u, y, self.Q, self.R, self.f, self.g, self.A, self.C, self.iEKF_maxIter, self.iEKF_eps)
            elif self.filter.__name__.lower() == "pf":
                # Note that f and g need to be specific to PF!
                mu, sigma = self.filter(X, u, y, self.Q, self.R, self.f, self.g, self.PF_numParticles, self.PF_resample)
            else:
                raise Exception("Invalid filter name")
            # Store
            self.muHistory[:,i] = mu
            self.sigmaHistory[:,:,i] = sigma
        
        print("Filtering complete")


class GAInitialization:
    def __init__(self, mu0_state, sigma0_state, mu0_chromosome, sigma0_chromosome, nTimesteps, 
                 pop_size, selection_fxn, crossover_fxn, mutation_fxn, k_max, 
                 k_selection=None, L_interp_crossover=0.5, mutation_stdev=1, regularization_scaling=0):

        # Storing information from inputs
        self.pop_size = pop_size
        self.mu0_chromosome = mu0_chromosome
        self.sigma0_chromosome = sigma0_chromosome
        self.mu0_state = mu0_state
        self.sigma0_state = sigma0_state
        self.selection_fxn = selection_fxn
        self.crossover_fxn = crossover_fxn
        self.mutation_fxn = mutation_fxn
        self.k_max = k_max
        self.k_selection = k_selection
        self.L_interp_crossover = L_interp_crossover
        self.nTimesteps = nTimesteps
        self.mutation_stdev = mutation_stdev
        self.regularization_scaling = regularization_scaling
        self.population = self.getInitialPopulation()
        self.muHistories, self.sigmaHistories = self.initializeStorage()
        
        # Error checking on inputs
        if self.selection_fxn.__name__.lower() in {"truncationselection", "tournamentSelection"} and self.k_selection == None:
            raise Exception("Must specify k for truncation/tournament selection!")

    def getInitialPopulation(self):
        '''
        Given an initial guess for the filter parameters (mean_chromosome) and the 
        covariance in this estimate (cov_chromosome), sample pop_size possible
        filters to use for the initial population
        - Return shape: list of length pop_size, containing chromosomes
        '''
        return [np.random.multivariate_normal(self.mu0_chromosome, self.sigma0_chromosome) for _ in range(self.pop_size)]
        # return np.random.multivariate_normal(self.mean_chromosome, self.cov_chromosome, size=self.pop_size).T
    
    def initializeStorage(self):
        '''
        Returns two lists, each of length pop_size, containing the muHistory, sigmaHistory for each filter
        '''
        dim = len(self.mu0_state)
        muHistory = np.zeros((dim, self.nTimesteps))
        sigmaHistory = np.zeros((dim, dim, self.nTimesteps))
        muHistory[:,0] = self.mu0_state
        sigmaHistory[:,:,0] = self.sigma0_state
        return [copy.deepcopy(muHistory) for _ in range(self.pop_size)], [copy.deepcopy(sigmaHistory) for _ in range(self.pop_size)]

# Assumes we are using the kalman filter
class GAResult:
    def __init__(self, filter, problem, measurement_history, GA_initialization):
        self.Q = problem.noise_model.Q
        self.R = problem.noise_model.R
        self.f = problem.dynamics_model.f
        # self.A = problem.dynamics_model.A UNKNOWN!!!!
        self.g = problem.measurement_model.g
        self.C = problem.measurement_model.C
        self.uHistory = problem.controls_model.uHistory
        self.yHistory = measurement_history
        self.filter = filter
        self.nTimesteps = problem.time_model.nTimesteps

        # Note: check to see if you moved all of the parameters over!!!
        self.muHistories = GA_initialization.muHistories # Initial
        self.sigmaHistories = GA_initialization.sigmaHistories # Initial
        self.mu0_state = GA_initialization.mu0_state
        self.sigma0_state = GA_initialization.sigma0_state
        self.population = GA_initialization.population
        self.pop_size = GA_initialization.pop_size
        self.selection_fxn = GA_initialization.selection_fxn
        self.crossover_fxn = GA_initialization.crossover_fxn
        self.mutation_fxn = GA_initialization.mutation_fxn
        self.k_max = GA_initialization.k_max
        self.k_selection = GA_initialization.k_selection
        self.L_interp_crossover = GA_initialization.L_interp_crossover
        self.mutation_stdev = GA_initialization.mutation_stdev
        
        self.regularization_scaling = GA_initialization.regularization_scaling

        self.bestSoFar = None
        self.bestHistory = []

        self.evalHistory = []

        # Keeping track of the GA results at each iteration for plotting later on
        # Will be a list containing the population at each iteration
        self.populationHistory = [self.population] # Initialize! Append after each iteration
        
        # Also want to keep track of the full trajectory for plotting purposes
        self.muHistories_history = [] # Initially empty, before any filters are run
        
        # Run the genetic algo
        self.geneticAlgorithm()
        
        # Before we terminate, evaluate the filters one last time and choose the single best
        evals = self.evaluatePopulation()
        # bestSoFar will get updated when this is run
        print(f"Final result: {self.bestSoFar}")

    def chromosomeToA(self, chromosome):
        '''
        Converts a chromosome array to a square A matrix for the KF, returns the matrix
        '''
        length = len(chromosome)
        n = np.sqrt(length)
        if np.mod(n, 1) != 0:
            raise Exception(f"The chromosome length is not a square number! Length: {length}")
        return np.reshape(chromosome, (int(n), int(n)))
        
    def runFilters(self):
        '''
        - Takes each chromosome, turns it into an A matrix, and then uses this to run the 
          Kalman Filter over the full duration of the simulation measurement history
        => Does not return. Saves information to self via muHistories, sigmaHistories,
           and muHistories_history
        '''
        # Filter each chromosome for all timesteps
        for chromo_ind in range(self.pop_size):
            # Initialize mu and sigma to the mu0, sigma0 values stored
            mu = self.mu0_state
            sigma = self.sigma0_state
            # Get A based on the current chromosome
            chromosome = self.population[chromo_ind]
            A = self.chromosomeToA(chromosome)
            for i in range(1, self.nTimesteps):
                # Retrieve from simulation data
                y = self.yHistory[:,i]
                u = self.uHistory[:,i]
                # Filter, using the Kalman filter
                mu, sigma = self.filter(mu, sigma, u, y, self.Q, self.R, self.f, self.g, A, self.C)
                # Store
                self.muHistories[chromo_ind][:,i] = mu
                self.sigmaHistories[chromo_ind][:,:,i] = sigma
        # Save our data
        self.muHistories_history.append(copy.deepcopy(self.muHistories))
        print("Filtering complete")
    
    def evaluatePopulation(self):
        '''
        - To evaluate the population of possible unpacked A matrices (the chromosomes), we need to run the filters 
          on each one to have a trajectory to evaluate
        - For each chromosome, go through the trajectory and evaluate the measurement likelihood at each point
        - Use the Mahalanobis distance to determine performance - goal is to minimize
        => Returns the sum of the Mahalanobis distances for each filter as the evaluation metric for each
        '''
        self.runFilters()
        evals = []
        # For each chromosome, go through the muhistory at each timestep and determine
        # the measurement likelihood 
        for chromo_ind in range(self.pop_size):
            muHistory = self.muHistories[chromo_ind]
            sigmaHistory = self.sigmaHistories[chromo_ind]
            metric = 0 # Initialize
            for i in range(self.nTimesteps):
                y = self.yHistory[:,i]
                sigma = sigmaHistory[:,:,i]
                mu = muHistory[:,i]
                mahalanobis_dist = (y - self.g(mu)) @ np.linalg.inv(self.C @ sigma @ self.C.T + self.R) @ (y - self.g(mu))
                # Minimizing the sum of mahalanobis distances is equivalent to minimizing the negative log likelihood, 
                # which is also equivalent to minimizing the product of probabilities (but more numerically stable)
                metric += mahalanobis_dist
                metric += self.regularization_scaling * np.sum(abs(self.population[chromo_ind]))
            evals.append(metric)
        
        # Save data on the best we've seen
        self.bestSoFar = self.population[np.argmin(evals)]
        self.bestHistory.append(self.bestSoFar)
        self.evalHistory.append(evals)
        return evals


    def geneticAlgorithm(self):
        '''
        - Run the GA for k_max iterations. Every time we get a new population, we need to
          re-evaluate their performance, which involves re-filtering with the new matrix
        - Uses the selection, crossover, and mutation functions as defined in the initialization
        => Does not return. Updates the population in place and stores it in populationHistory
        '''
        for k in range(self.k_max):
            evals = self.evaluatePopulation()            
            parents = self.selection_fxn(evals, self.k_selection)
            children = [self.crossover_fxn(self.population[p[0]], self.population[p[1]]) for p in parents]
            self.population = [self.mutation_fxn(child, self.mutation_stdev) for child in children]
            self.populationHistory.append(self.population) # Save data
            print("GA iteration complete")
        print("All GA iterations complete")