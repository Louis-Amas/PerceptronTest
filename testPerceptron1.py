import matplotlib.pyplot as plt
import numpy as np

from Perceptron import Perceptron


def drawLine(perceptron):
    weights = perceptron.getWeights()
    w0 = weights[0]
    w1 = weights[1]
    pt1 = -w0 * -1 / w1
    pt2 = -w0 * 1 / w1
    plt.plot( [ -1, 1 ] , [ pt1 , pt2 ] )

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


p = Perceptron(datas[0].shape)

p.train(datas, targets)

guess = p.guess(datas)

scores = guess - targets[:,0]
drawLine(p)


plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()
