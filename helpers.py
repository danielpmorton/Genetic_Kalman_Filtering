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


# Plotting
def makePlot(times, xdata, mudata, sigmadata, testName):
    plt.plot(times, xdata, c='b')
    plt.plot(times, mudata, c='r')
    upper = mudata + 1.96*np.sqrt(sigmadata)
    lower = mudata - 1.96*np.sqrt(sigmadata)
    plt.fill_between(times, upper, lower, alpha=0.3, color='r')
    plt.legend(["Ground Truth", "Filtered Result", "95% Confidence Interval"])
    plt.xlabel("Time")
    plt.ylabel(testName)

def makeXYPlot(x, y, mux, muy, title):
    plt.plot(x, y, c='b')
    plt.plot(mux, muy, c='r')
    plt.legend(["Ground Truth", "Filtered Result"])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')

def plotFilter(times, xHistory, muHistory, sigmaHistory, title):
    xdim = xHistory.shape[0]

    x1data, x2data, x3data = [xHistory[i,:] for i in range(xdim)]
    mu1data, mu2data, mu3data = [muHistory[i,:] for i in range(xdim)]
    sigma11data, sigma22data, sigma33data = [sigmaHistory[i,i,:] for i in range(xdim)]

    plt.subplot(3,1,1)
    makePlot(times, x1data, mu1data, sigma11data, 'px') # Update these based on the actual names of the state variables

    plt.subplot(3,1,2)
    makePlot(times, x2data, mu2data, sigma22data, 'py')

    plt.subplot(3,1,3)
    makePlot(times, x3data, mu3data, sigma33data, 'theta')

    plt.suptitle(title)
    plt.show()

    makeXYPlot(x1data, x2data, mu1data, mu2data, title)
    plt.show()

def plotSLAM(groundTruthHistory, slamHistory, title, numMapPoints=30, dotSize=10):
    px_slam = slamHistory[0,:]
    py_slam = slamHistory[1,:]
    map_slam = slamHistory[3:,:]
    px_truth = groundTruthHistory[0,:]
    py_truth = groundTruthHistory[1,:]
    map_truth = groundTruthHistory[3:,0] # Only need one timestep because stationary
    nTimesteps = slamHistory.shape[1]
    # Plotting the pose result
    plt.plot(px_truth, py_truth, c='b', label="Ground Truth Trajectory")
    plt.plot(px_slam, py_slam, c='r', label="Filtered Trajectory")
    # Plotting the mapping result
    # Randomly sampling a number of map filtering results based on numPoints
    selectedIdxs = np.random.randint(nTimesteps, size=numMapPoints)
    selected = map_slam[:,selectedIdxs]
    map_x_slam = selected[::2,:]
    map_y_slam = selected[1::2,:]
    map_x_truth = map_truth[::2]
    map_y_truth = map_truth[1::2]
    plt.scatter(map_x_slam, map_y_slam, dotSize, c='r', label="Filtered Map")
    plt.scatter(map_x_truth, map_y_truth, dotSize*3, c='b', label="Ground Truth Map")
    # Plot config
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()

def plot_SMD_OLD(times, xHistory, muHistory, sigmaHistory, title, ylabels):
    xdim = xHistory.shape[0]

    for i in range(xdim):
        plt.subplot(xdim, 1, i+1)
        makePlot(times, xHistory[i,:], muHistory[i,:], sigmaHistory[i,i,:], ylabels[i])
    
    plt.suptitle(title)
    plt.show()

def makePlot_new(times, xdata, mudata, sigmadata, testName, ax):
    ax.plot(times, xdata, c='b', label="Ground Truth")
    ax.plot(times, mudata, c='r', label="Filtered Result")
    upper = mudata + 1.96*np.sqrt(sigmadata)
    lower = mudata - 1.96*np.sqrt(sigmadata)
    ax.fill_between(times, upper, lower, alpha=0.3, color='r', label="95% Confidence Interval")
    ax.set_xlabel("Time")
    ax.set_ylabel(testName)


def plot_SMD(times, xHistory, muHistory, sigmaHistory, title, ylabels):
    xdim = xHistory.shape[0]
    fig, axes = plt.subplots(xdim, 1)

    for i in range(xdim):
        plt.subplot(xdim, 1, i+1)
        makePlot_new(times, xHistory[i,:], muHistory[i,:], sigmaHistory[i,i,:], ylabels[i], axes[i])
    
    plt.subplots_adjust(right=0.8)

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc="center right") # , bbox_to_anchor=(1, 0.5)

    plt.suptitle(title)
    plt.show()


def plotGA():
    # NOTE: probably also want to keep track of the results of the GA at each iteration
    # The idea would be to plot this history and show how the GA evolved
    # What the plot would look like:
    # 1 subplot for every entry in the A matrix (if A is 2x2, then have 4 subplots)
    # - X axis: each iteration of the GA
    # - Y axis: the magnitude of the relevant component of A
    # - At each iteration, scatter the values from every chromosome in the population

    raise Exception("Not implemented yet!")