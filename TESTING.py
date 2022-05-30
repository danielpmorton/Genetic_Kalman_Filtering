import numpy as np

def rouletteWheelSelection(y):
    fitnesses = np.max(y) - y
    fitnesses = fitnesses / np.linalg.norm(fitnesses, ord=1) # L1 normalize the fitnesses
    return [np.random.choice(np.arange(len(y)), size=2, p=fitnesses) for _ in range(len(y))]

y = np.array([2,4,7,3,2,5,7,4,1])
k = 4
test = rouletteWheelSelection(y)
print(test)
