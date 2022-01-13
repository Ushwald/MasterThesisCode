import numpy as np

"""This code will run a simulation based on gradient descent, to obtain curves 
for generalization error for varying types of student-teacher scenarios.


We will need:

- A classifier interface supporting training until some specified form of convergence, generalization error
	- A teacher implementation
	- Various student implementations

- A dataset, distributed as we like but as a starting point Gaussian

"""

class BinNPC:
    
    def __init__(self, N: int):
        # initialize a random MNPC with M clusters in the layer
        self.N = N
        # Initialize NPC paremeters however you wish. Actual restrictions on NPC parameters should also go into the training function
        # Normalize W, and experiment with restrictions on omega
        self.threshold = 0.0
        self.W = (np.random.rand(N) - 0.5) * 2
        self.W = self.W / np.linalg.norm(self.W)# * N 
        self.omega_i = (np.random.rand(N, N) - 0.5)# * N 
        for i in range(N):
            self.omega_i[i, i] = 0
            for j in range(i):
                self.omega_i[i, j] = self.omega_i[j, i]

    def getActivation(self, example):
        activation =  np.dot(self.W, example)
        for i in range(self.N):
            for j in range(self.N):
                activation += self.omega_i[i, j] * example[i] * example[j]
        activation += self.threshold
        return activation


    def getActivations(self, examples):
        activations = []
        for example in np.transpose(examples):
            activations.append(self.getActivation(example))
        return activations

    def train(self, examples, labels, T, stop = 0.05, rate = 0.01):	
    	# Gradient descent plus white noise at temperature T, 
    	# stop after average training error improvement is below stop
    	trainingError = 0.0
    	errorDelta = 0.0
    	while True:
    		# Train
    		for exampleIdx, example in enumerate(np.transpose(examples)): #(The following is just a rough draft)
    			self.W -= 			example * np.sign(self.getActivation(example) * label[exampleIdx]) * rate + np.random.normal(0.0, T, (shape(self.W)))
    			self.omega_i -= 	example * example.T * np.sign(self.getActivation(example) * label[exampleIdx]) * rate np.random.normal(0.0, T, (shape(self.omega_i)))

    			self.W = self.W / np.sqrt(np.linalg.norm(self.W)**2 + np.linalg.norm(np.flatten(self.omega_i))**2)
    			self.omega_i = self.W / np.sqrt(np.linalg.norm(self.W)**2 + np.linalg.norm(np.flatten(self.omega_i))**2)
    		if errorDelta < stop:
    			break;


def getInputs(N, p, distribution = 'normal'):
    if distribution == 'normal':
        ret = np.random.normal(0.0, 1.0, (N, p)).transpose()
        return (ret - np.average(ret)).transpose() #Centered data
    elif distribution == 'hypercube':
        ret = np.random.uniform(-1.0, 1.0, (N, p)).transpose()
        return (ret - np.average(ret)).transpose() #Centered data
    elif distribution == 'hypersphere':
        ret = np.random.uniform(-1.0, 1.0, (N, p)).transpose()
        for pidx, point in enumerate(ret):
            while np.linalg.norm(ret[pidx]) > 1.0:
                ret[pidx] = np.random.uniform(-1.0, 1.0, (N))
        return (ret - np.average(ret)).transpose()


