import numpy as np
import matplotlib.pyplot as plt

from Perceptron import Perceptron


def AddOne(points):
    newPoints = []
    for point in points:
        newPoints.append(np.append(point, 1))
    return np.array(newPoints)

def computePt(x0, weights, bias=0):
    w0 = weights[0]
    w1 = weights[1]
    x1 = ( - w0 * x0 - bias ) / w1
    return np.array([x0, x1])


def drawLine(pt1, pt2):
    plt.plot( [pt1[0], pt2[0]] , [pt1[0], pt2[0]]   , 'ro', color= 'g')



def generateData3D(n):
    """
    generates a 2D linearly separable dataset with 2n samples.
    """
    X = (2*np.random.rand(2*n,2)-1)/2 - 0.5
    X[:n,0] += 1
    X[n:,0] += 2
    X[:n,1] += 0.5
    Y = np.ones([2*n,1])
    Y[n:] -= 2
    return X,Y

datas = generateData3D(5)
points = datas[0]
points = AddOne(points)
targets = datas[1]

bluePoints = []
redPoints  = []

for i, point in enumerate(points):
    if targets[i] == 1:
        bluePoints.append(point)
    else:
        redPoints.append(point)

bluePoints = np.array(bluePoints)
redPoints = np.array(redPoints)


perceptron = Perceptron(points[0].shape, 0.1)

perceptron.train(points, targets)
guess = []
for point in points:
    guess.append(perceptron.guess(point))

guess = np.array(guess)
scores = targets[:,0] - guess

max = np.amax(points[:,0])
min = np.amin(points[:,0])


pt1 = computePt(min, perceptron.getWeights(),1)
pt2 = computePt(max, perceptron.getWeights(),1)
plt.plot(bluePoints[:,0], bluePoints[:,1], 'ro', color='b')
plt.plot(redPoints[:,0], redPoints[:,1], 'ro', color='r')

drawLine(pt1, pt2)

plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()
