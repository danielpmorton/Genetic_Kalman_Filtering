import scipy.linalg
import numpy as np

# From equation 6 in "Ekf modifications from an optimization viewpoint"
# Seems to be like an objective function?
def V(x, R, sigma, y, g, f, u):
    r = np.array([[np.linalg.inv(scipy.linalg.sqrtm(R)) @ (y - g(x))], 
                  [np.linalg.inv(scipy.linalg.sqrtm(sigma)) @ (f(x,u) - x)]])
    return (1/2) * r.T @ r