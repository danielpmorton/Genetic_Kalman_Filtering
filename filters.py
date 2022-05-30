import numpy as np
from scipy.stats import multivariate_normal
import random
from classes import SigmaPoints, ParticleSet

# A and C must be linear/constant to apply this
def KF(mu, sigma, u, y, Q, R, f, g, A, C):
    # Predict
    mubar = f(mu, u)
    sigmabar = A @ sigma @ A.T + Q
    # Calculate Kalman gain
    K = sigmabar @ C.T @ np.linalg.inv(C @ sigmabar @ C.T + R)
    # Update
    mu = mubar + K @ (y - g(mubar))
    sigma = sigmabar - K @ C @ sigmabar
    return mu, sigma


def EKF(mu, sigma, u, y, Q, R, f, g, get_A, get_C):
    # Predict
    A = get_A(mu, u)
    mubar = f(mu, u)
    sigmabar = A @ sigma @ A.T + Q
    # Calculate Kalman gain
    C = get_C(mubar)
    K = sigmabar @ C.T @ np.linalg.inv(C @ sigmabar @ C.T + R)
    # Update
    mu = mubar + K @ (y - g(mubar))
    sigma = sigmabar - K @ C @ sigmabar
    return mu, sigma

def iEKF(mu, sigma, u, y, Q, R, f, g, get_A, get_C, maxIterations, eps):
    # Predict
    A = get_A(mu, u)
    mubar = f(mu, u)
    sigmabar = A @ sigma @ A.T + Q
    # Iterative update step
    converged = False
    mu_i = mubar
    mu_i_prev = mubar * 1e5 # Random large value
    iter = 0
    while not converged and iter < maxIterations:
        C_i = get_C(mu_i)
        K_i = sigmabar @ C_i.T @ np.linalg.inv(C_i @ sigmabar @ C_i.T + R)
        mu_i = mubar + K_i @ (y - g(mu_i)) + K_i @ C_i @ (mu_i - mubar)
        if np.linalg.norm(mu_i - mu_i_prev) < eps:
            converged = True
        else:
            mu_i_prev = mu_i
            iter += 1
    mu = mu_i
    sigma = sigmabar - K_i @ C_i @ sigmabar
    return mu, sigma


def UKF(mu, sigma, u, y, Q, R, f, g, L):

    def apply_f_to_X(X, u):
        X.points = [f(point, u) for point in X.points]

    def get_mubar(X):
        return sum([weight * point for point, weight in zip(X.points, X.weights)])

    def get_sigmabar(X, mubar, Q):
        return sum([weight * np.outer((point-mubar), (point-mubar)) for point, weight in zip(X.points, X.weights)]) + Q

    def get_mu_and_sigma(Xbar, mubar, sigmabar, y):
        yhat = sum([weight*g(point) for point, weight in zip(Xbar.points, Xbar.weights)])
        sigma_y = sum([weight * np.outer((g(point)-yhat), (g(point)-yhat)) for point, weight in zip(Xbar.points, Xbar.weights)]) + R
        sigma_xy = sum([weight * np.outer((point-mubar), (g(point)-yhat)) for point, weight in zip(Xbar.points, Xbar.weights)])
        syinv = np.linalg.inv(sigma_y)
        sigma = sigmabar - sigma_xy @ syinv @ sigma_xy.T
        mu = mubar + sigma_xy @ syinv @ (y - yhat)
        return mu, sigma

    # Predict
    X = SigmaPoints(mu, sigma, L) # Get sigma points based on current mu, sigma 
    apply_f_to_X(X, u) # Propagate sigma points through dynamics 
    mubar = get_mubar(X)
    sigmabar = get_sigmabar(X, mubar, Q)
    # Update
    Xbar = SigmaPoints(mubar, sigmabar, L)
    mu, sigma = get_mu_and_sigma(Xbar, mubar, sigmabar, y)
    return mu, sigma

def PF(X, u, y, Q, R, f_PF, g_PF, numParticles, resample=True):

    def update_weights(X, y):
        # Evaluate g on each of the particles, so that we can compare the expected and received measurement
        gEvals = g_PF(X)
        # Get the probablity distribution values: Evaluate y when the mean is g(particle)
        whats = multivariate_normal.pdf(y, gEvals, R) 
        X.weights = whats / np.sum(whats) # Normalize
        
    def importanceResample(X):
        selectedInds = random.choices(np.arange(numParticles), weights=X.weights, k=numParticles) # Get a random selection of indices
        X.particles = X.particles[:,selectedInds]
        X.weights = np.ones(numParticles) / numParticles # Reset the weights

    # Predict
    f_PF(X, u)
    # Update
    update_weights(X, y)
    if resample: # Change the degault value if you don't want to resample every iteration
        importanceResample(X) 
    mu = np.average(X.particles, weights=X.weights, axis=1) 
    sigma = np.cov(X.particles)
    return mu, sigma 



# THIS DOESN'T MAKE SENSE !!!!
# Can't apply the AA222 algorithm here???
def BLS_iEKF(mu, sigma, u, y, Q, R, f, g, get_A, get_C, maxIterations, d, alpha, beta=1e-4, p=0.5):
    # Predict
    A = get_A(mu, u)
    mubar = f(mu, u)
    sigmabar = A @ sigma @ A.T + Q
    # Iterative update step
    converged = False
    mu_i = mubar
    mu_i_prev = mubar * 1e5 # Random large value
    iter = 0

    # Get an initial gradient
    C_i = get_C(mu_i)
    K_i = sigmabar @ C_i.T @ np.linalg.inv(C_i @ sigmabar @ C_i.T + R)
    orig_gx = K_i @ (y - g(mu_i)) + K_i @ C_i @ (mu_i - mubar)
    
    orig_fx = objective_fxn(mubar) # This is equivalent to "y" in the AA222 notes

    while not converged and iter < maxIterations:
        if objective_fxn(mubar + alpha*d) < orig_fx + beta * alpha * np.dot(orig_gx,d):
            converged = True
        else:
            alpha *= p
            iter += 1


    # Apply the converged step length to mu
    mu = mubar + alpha * d
    sigma = sigmabar - K_i @ C_i @ sigmabar
    return mu, sigma

