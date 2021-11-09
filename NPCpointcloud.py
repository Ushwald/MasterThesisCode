from perceptron import BinPerceptron
from binaryClassifier import BinaryClassifier
import numpy as np
import random
import matplotlib.pyplot as plt

class BinNPC:
    """This version of the NPC takes in symmetric binary inputs (-1 or 1)
    It works simply by implicitly defining an N^2 perceptron and assigning the appropriate N^2 weights"""

    def __init__(self, N: int):
        # initialize a random MNPC with M clusters in the layer
        self.N = N
        # Initialize NPC paremeters however you wish. Actual restrictions on NPC parameters should also go into the training function
        # Normalize W, and experiment with restrictions on omega
        self.W = (np.random.rand(N) - 0.5) * 2
        self.W = self.W / np.linalg.norm(self.W)
        self.omega_i = (np.random.rand(N, N) - 0.5) * 2 
        for i in range(N):
            self.omega_i[i, i] = 0
            for j in range(i):
                self.omega_i[i, j] = self.omega_i[j, i]

    def getActivations(self, examples):
        activations = []
        for example in np.transpose(examples):
            activation =  np.dot(self.W, example)
            for i in range(self.N):
                for j in range(self.N):
                    activation += self.omega_i[i, j] * example[i] * example[j]
            activations.append(activation)
        return activations

def getX(N):
    return np.random.normal(0.0, 1.0, N)


N = 8
p = 100000
alpha_val = min(1.0 / p * 10000, 1.0)
npc1 = BinNPC(N)
npc2 = BinNPC(N)

# Create a point cloud plot of u, \underscore u activations for gaussian distributed inputs

points = np.random.normal(0.0, 1.0, (N, p))
activations1 = npc1.getActivations(points)
activations2 = npc2.getActivations(points)

fig = plt.figure(figsize=(8, 8), dpi=80)
axes = plt.gca()

plt.scatter(activations1, activations2, color = 'red', alpha = alpha_val)
plt.scatter(points[0], points[1], color = 'blue', alpha = alpha_val)


axislim = [0.7 * min(axes.get_xlim()[0], axes.get_ylim()[0]),0.7 * max(axes.get_xlim()[1], axes.get_ylim()[1])]
axes.set_xlim(axislim)
axes.set_ylim(axislim)




plt.show()
