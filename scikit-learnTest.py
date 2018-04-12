import numpy as np
from sklearn import linear_model

def generateData(n):
    """
    generates a 2D linearly separable dataset with 2n samples.
    """
    X = (2*np.random.rand(2*n,2)-1)/2 - 0.5
    X[:n,1] += 1
    X[n:,0] += 1
    Y = np.ones([2*n,1])
    Y[n:] -= 2
    return X,Y


data = generateData(50)
target = data[1][:,0]
points = data[0]


p = linear_model.Perceptron(max_iter=500)

p.fit(points, target)

res = p.predict(points)
