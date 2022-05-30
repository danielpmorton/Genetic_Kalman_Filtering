import numpy as np
import random

# NOTE All genetic algos have been moved to the other python file

# From AA222 textbook, converted to Python
def backtracking_line_search_AA222(f, gradf, x, d, alpha, p=0.5, beta=1e-4):
    '''
    f: The objective function
    gradf: The gradient of the objective function
    x: The current design point
    d: A descent direction
    alpha: A maximum step size
    p: Reduction factor
    beta: First Wolfe condition parameter
    '''
    y = f(x)
    g = gradf(x)
    while f(x + alpha*d) > y + beta * alpha * np.dot(g,d):
        alpha *= p
    return alpha