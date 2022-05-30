import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

'''
What do we want to show with the plots?
    How the genetic algorithm evolves over time

- What do the filtered trajectories look like for each in the population?
    Compare these to the ground truth and the best seen
- How do the values in the population evolve over time
    Similarly, want to highlight the best one so far and how this changes
- Convergence plot for mahalanobis distance metric evaluations

NOTE: This entire file assumes that we're working with the Spring-Mass-Damper system
    So, we have a state of dimension 2 - position and velocity
'''

# Plotting the single best trajectory versus the ground truth
def plot_best_filter_vs_truth(times, xHistory, muHistory, sigmaHistory):
    '''
    times:        Array of times.           Shape (nTimesteps,)
    xHistory:     Ground-truth information. Shape (xdim, nTimesteps)
    muHistory:    Filtered pose estimate.   Shape (xdim, nTimesteps)
    sigmaHistory: Filtered covariance.      Shape (xdim, xdim, nTimesteps)
    '''
    # Helper function
    def makeSubplot(times, xdata, mudata, sigmadata, testName):
        plt.plot(times, xdata, c='k')
        plt.plot(times, mudata, c='r')
        upper = mudata + 1.96*np.sqrt(sigmadata)
        lower = mudata - 1.96*np.sqrt(sigmadata)
        plt.fill_between(times, upper, lower, alpha=0.3, color='r')
        plt.legend(["Ground Truth", "Filtered Result", "95% Confidence Interval"])
        plt.xlabel("Time")
        plt.ylabel(testName)

    xdim = xHistory.shape[0]

    x1data, x2data = [xHistory[i,:] for i in range(xdim)]
    mu1data, mu2data = [muHistory[i,:] for i in range(xdim)]
    sigma11data, sigma22data, = [sigmaHistory[i,i,:] for i in range(xdim)]

    plt.subplot(2,1,1)
    makeSubplot(times, x1data, mu1data, sigma11data, 'Position')

    plt.subplot(2,1,2)
    makeSubplot(times, x2data, mu2data, sigma22data, 'Velocity')

    plt.suptitle("Comparing the best filter to the ground truth")
    plt.show()

# Plot a whole generation of the algorithm
def plot_generation(times, xHistory, muHistories_history, evalHistory, gen=0):
    '''
    Plots each XY trajectory with variable color depending on the weight

    xHistory: Ground truth information
    muHistories_history: 
        muHistories: A saved set of trajectories for each chromosome in a generation
                     List of length (pop_size) with each entry = a muHistory, shape (xdim, nTimesteps)
    evalHistory: 
        evals: The mahalanobis distance metrics for each of the filters in muHistories
    '''
    # Load the info from the desired generation
    muHistories = muHistories_history[gen]
    evals = evalHistory[gen]

    # Plotting the ground truth
    plt.plot(times, xHistory[0,:], c='k', label="Ground Truth")

    # Plotting all trajectories with variable color, semi-transparent
    # num_filters = len(evals)
    # ranks = rankdata(evals)
    for (i, muHistory) in enumerate(muHistories):
        # rank_normalized = ranks[i] / num_filters
        plt.plot(times, muHistory[0,:], c='r', alpha=0.2, label="__nolegend__")
    
    # Plotting the best filter in a distinguishable color
    best_muHistory = muHistories[np.argmin(evals)]
    plt.plot(times, best_muHistory[0,:], c='b', alpha=0.7, label="Best filter")

    plt.title("Comparison of filters within a generation")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.legend()
    # plt.gca().set_aspect('equal', adjustable='box')   
    plt.show()
    
def plot_mahalanobis_convergence(evalHistory):
    '''
    Shows how the mahalanobis values for all of the filters evolves with each iteration of the GA

    '''
    iterations = np.arange(len(evalHistory))
    pop_size = len(evalHistory[0])
    evalHistory_array = np.array(evalHistory)
    # Plotting the evaluation history for every filter
    for i in range(pop_size):
        plt.plot(iterations, evalHistory_array[:,i], c='r', alpha=0.3, label="All evaluations")
    # Plotting the evaluation history for the best at every iteration
    # bestEvals = []
    # for i in range(len(evalHistory)):
    #     evals = evalHistory[i]
    #     bestEvals.append(evals[np.argmin(evals)])
    # plt.plot(iterations, bestEvals, c='k', label="Best evaluations")
    
    plt.title("Loss Function Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Evaluation value")
    # plt.legend()
    plt.yscale('log')
    plt.show()

def plot_chromosome_convergence(bestHistory, trueChromosome):
    '''
    Shows how each value in the chromosome changes over time
    Uses the best filter (chromosome) at each iteration to show this
    '''
    iterations = np.arange(len(bestHistory))
    chromosome_size = len(trueChromosome)
    bestHistory_array = np.array(bestHistory)
    for i in range(chromosome_size): 
        plt.subplot(chromosome_size, 1, i+1)
        plt.plot(iterations, trueChromosome[i]*np.ones(len(iterations)), c='b', label="True value")
        plt.plot(iterations, bestHistory_array[:,i], c='r', label="GA result")
        plt.xlabel("Iteration")
        plt.ylabel(f"Bit {i}")
    
    plt.suptitle("Chromosome Value Convergence")
    plt.legend()
    plt.show()

def plot_optimal_filter_vs_truth(times, xHistory, muHistory, sigmaHistory):
    '''
    times:        Array of times.           Shape (nTimesteps,)
    xHistory:     Ground-truth information. Shape (xdim, nTimesteps)
    muHistory:    Filtered pose estimate.   Shape (xdim, nTimesteps)
    sigmaHistory: Filtered covariance.      Shape (xdim, xdim, nTimesteps)
    '''
    # Helper function
    def makeSubplot(times, xdata, mudata, sigmadata, testName):
        plt.plot(times, xdata, c='k')
        plt.plot(times, mudata, c='r')
        upper = mudata + 1.96*np.sqrt(sigmadata)
        lower = mudata - 1.96*np.sqrt(sigmadata)
        plt.fill_between(times, upper, lower, alpha=0.3, color='r')
        plt.legend(["Ground Truth", "Filtered Result", "95% Confidence Interval"])
        plt.xlabel("Time")
        plt.ylabel(testName)

    xdim = xHistory.shape[0]

    x1data, x2data = [xHistory[i,:] for i in range(xdim)]
    mu1data, mu2data = [muHistory[i,:] for i in range(xdim)]
    sigma11data, sigma22data, = [sigmaHistory[i,i,:] for i in range(xdim)]

    plt.subplot(2,1,1)
    makeSubplot(times, x1data, mu1data, sigma11data, 'Position')

    plt.subplot(2,1,2)
    makeSubplot(times, x2data, mu2data, sigma22data, 'Velocity')

    plt.suptitle("Results from the optimal filter")
    plt.show()