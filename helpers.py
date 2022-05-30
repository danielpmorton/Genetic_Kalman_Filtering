import numpy as np
import matplotlib.pyplot as plt

def seedRNG(seed):
    np.random.seed(seed)

def checkObservability(A, C):
    # Note: Can evaluate this at any timestep, any number of times
    dim = A.shape[0]
    O_pt2 = np.squeeze(np.array([C@np.linalg.matrix_power(A, i) for i in range(1, dim)]))
    O = np.vstack((C, O_pt2))
    rank = np.linalg.matrix_rank(O)
    if rank == dim:
        print("Observable")
    else:
        print(f"Not observable. Rank = {rank} and we need {dim}")
