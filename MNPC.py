from perceptron import BinPerceptron
from binaryClassifier import BinaryClassifier
import numpy as np
import random

class BinMNPC(BinaryClassifier):
	"""The binary MNPC will be built essentially as a perceptron 
		to be trained on a dataset which is transformed already by the 
		NPC layer"""
	def __init__(self, M: int, N: int):
		# initialize a random MNPC with M clusters in the layer
		self.N = N
		self.M = M
		# While the following would create randomized omegas, I found out it didn't seem to converge for these.
		self.omega_i = np.random.rand(M, N ** 2) 
		# Therefore I replace them now, for debugging purposes, with a setup that should be equivalent to 
		# just a perceptron (assuming N == M, with each of the NPC simply returning one input
		#self.omega_i = np.zeros(shape = (M, N**2))
		#for omegaIdx, omega in enumerate(self.omega_i):
		#	omega[omegaIdx * N + omegaIdx] = 1.0
		self.perceptron = BinPerceptron(M) # The perceptron takes inputs from each of the M NPC's

	def transformInput(self, example: np.mat):
		return 	np.array([np.dot(np.matmul(example.T,  example).flatten(), 
			  	self.omega_i[i])[0, 0] for i in range(self.M)])

	def label(self, example):
		return 	self.perceptron.label(self.transformInput(np.mat(example)))



	def train(self, examples, labels):
		# Train the perceptron on the transformed dataset:
		# So first transform the dataset:

		transformedExamples = np.ndarray(shape = (examples.shape[0], self.M))
		for exampleIdx, example in enumerate(examples):
			transformedExamples[exampleIdx] = self.transformInput(np.mat(example))

		while True:
			hasConverged = True # may be set to false if update was still required
			for exampleIdx, example in enumerate(transformedExamples):
				# We apply the perceptron training algorithm, as does the book
				if self.label(example) * labels[exampleIdx] <= 0:
					# Update the weight vector:
					self.perceptron.weights = self.perceptron.weights + example * labels[exampleIdx]/np.sqrt(self.N)
					hasConverged = False 
			if hasConverged: # The algorithm has converged
				break

	
	
	def getErr(self, examples, labels):
		successCount = 0

		for exampleIdx, example in enumerate(examples):
			if (self.label(example) == labels[exampleIdx]):
				successCount += 1
		return (1 - successCount / len(examples))



# temporary debuging stuff:


def sampleOneExample(N: int):
	# N is twice the length of the binary (-1, 1) encoded number
	return np.mat([random.randrange(2) * 2 - 1 for _ in range(N)])

def getTrainingSet(N: int, p: int):
	trainingSet = np.ndarray(shape = (p, N), dtype = int)
	for exampleIdx, _ in enumerate(trainingSet):
		trainingSet[exampleIdx] = sampleOneExample(N)
	return trainingSet

# checking if our MNPC is indeed equivalent of a perceptron with the current simplifications:
N = 3

ptron = BinPerceptron(N)
mnpc = BinMNPC(N, N)
mnpc.perceptron = ptron

example = sampleOneExample(N)
print(example)
print(np.matmul(example.T,  example).flatten())

print(mnpc.transformInput(example))
print(ptron.label(example))
print(mnpc.label(example))

