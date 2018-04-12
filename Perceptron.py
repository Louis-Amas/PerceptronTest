import numpy as np

class Perceptron():

    def sign(self, nb):
        if nb > 0:
            return 1
        else:
            return -1

    def __init__(self, inputsLen, alpha):
        self.weights = np.zeros(inputsLen)
        self.alpha = alpha

    def getWeights(self):
        return self.weights

    def guess(self, inputs):
        oldVal = 0
        for i, weight in enumerate(self.weights):
            oldVal += weight * inputs[i]
        return self.sign(oldVal)



    def train(self, inputs, targets):
        targets = targets[:,0]
        while(True):
            guess = []
            for i, input in enumerate(inputs):
                g = self.guess(input)
                guess.append(g)
                if g != targets[i]:
                    self.weights += (targets[i] * input) * self.alpha
            guess = np.array(guess)
            if np.array_equal(targets, guess):
                break



