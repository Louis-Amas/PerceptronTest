import matplotlib.pyplot as plt
import numpy as np

from Perceptron import Perceptron

def drawLine(pt1, pt2):
    plt.plot( [pt1[0], pt2[0]] , [pt1[0], pt2[0]]   , 'ro', color= 'g')

def computePt(x0, weights, bias=0):
    w0 = weights[0]
    w1 = weights[1]
    x1 = ( - w0 * x0 - bias ) / w1
    return np.array([x0, x1])


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


test = generateData(5)
datas = test[0]
targets = test[1]


bluePoints = []
redPoints  = []
for i, point in enumerate(datas):
    if targets[i] == 1:
        bluePoints.append(point)
    else:
        redPoints.append(point)

bluePoints = np.array(bluePoints)
redPoints = np.array(redPoints)

plt.plot(bluePoints[:,0], bluePoints[:,1], 'ro', color='b')
plt.plot(redPoints[:,0], redPoints[:,1], 'ro', color='r')


p = Perceptron(datas[0].shape, 0.1)

p.train(datas, targets)

pt1 = computePt(-1, p.getWeights())
pt2 = computePt(1, p.getWeights())
drawLine(pt1, pt2)

plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()
