import numpy as np

# Selection methods - to select m parental pairs for m chromosomes
def truncationSelection(y, k):
    '''
    - Selects random parent pairs out of the k best performers, 
      based on the evaluation values of y for each chromosome
    - Assumes that the goal is to MINIMIZE y
    - Returns a list of pairs. List is of the same length as y. 
      The values in the pairs are the indices corresponding to y
    - y: An array of eval function values for each chromosome
    - k: The number of top performers to choose from
    '''
    num_chromosomes = len(y)
    sorted_idxs = np.argsort(y)
    return [sorted_idxs[np.random.randint(k, size=2)] for _ in range(num_chromosomes)]

def tournamentSelection(y,k):
    '''
    Each parent is the fittest out of k randomly chosen 
    chromosomes from the population

    Returns a list of index pairs for the selected parents. Length = len(y)
    '''
    num_chromosomes = len(y)
    def getParent():
        random_idxs = np.random.randint(num_chromosomes, size=k)
        return random_idxs[np.argmin([y[i] for i in random_idxs])]
    return [np.array([getParent(), getParent()]) for _ in range(num_chromosomes)]
        

def rouletteWheelSelection(y, k=None):
    '''
    Selects each parent with probability proportional to its performance
    relative to the population

    Returns a list of index pairs for the selected parent indices
    '''
    fitnesses = np.max(y) - y
    # At initialization, all filters might all have the same evaluation, making fitnesses all 0.
    # So, instead of all 0, replace with all 1 so the division will work
    if all(val == 0 for val in fitnesses):
        fitnesses += 1 
    fitnesses = fitnesses / np.linalg.norm(fitnesses, ord=1) # L1 normalize the fitnesses
    return [np.random.choice(np.arange(len(y)), size=2, p=fitnesses) for _ in range(len(y))]


# Crossover Methods
def singlePointCrossover(a, b):
    '''
    First portion of parent A gets merged with the second portion of parent B, 
    with the crossover point determined randomly
    '''
    i = np.random.randint(len(a))
    return np.concatenate((a[:i], b[i:]))

def twoPointCrossover(a, b):
    '''
    A section of parent B gets spliced into a copy of parent A, with the start and 
    end points of the splice (the 2 crossover points) determined randomly
    '''
    n = len(a)
    i,j = np.random.randint(len(a), size=2)
    if i > j:
        (i,j) = (j,i)
    return np.concatenate((a[:i], b[i:j], a[j:]))

def uniformCrossover(a, b): 
    '''
    Each bit in the child has a 50/50 chance of coming from either parent
    '''
    child = a.copy()
    for i in range(len(a)):
        if np.random.rand() < 0.5:
            child[i] = b[i]
    return child

def interpolationCrossover(a, b, L=0.5):
    '''
    The child is a linear interpolation between the two parents, with lambda (L) 
    being a parameter determining the relative weighting (usually even weighting, L=0.5)
    '''
    return (1-L)*a + L*b
    
# Only a single viable mutation method for real-valued chromosomes
def gaussianMutation(chromosome, stdev=1): 
    '''
    Mutation via adding gaussian random noise over the chromosome, 
    with the noise distribution determined by stdev
    '''
    return chromosome + np.random.randn(len(chromosome))*stdev