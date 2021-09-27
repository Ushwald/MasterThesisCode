import numpy as np
from binaryClassifier import BinaryClassifier

class BinPerceptron(BinaryClassifier):
	
	def __init__(self, N):
		self.weights = np.random.rand(N) * 2 - 1 # initialize perceptron
		self.N = N

	def train(self, examples, labels):
		# Start with own weights, then continue training until converge on currently given examples
		while True:
			hasConverged = True #  If we insist on convergence, we may set this to false
			for exampleIdx, example in enumerate(examples):
				# We apply the perceptron training algorithm, as does the book
				if self.output(example) * labels[exampleIdx] <= 0:
					# Update the weight vector:
					self.weights = self.weights + example * labels[exampleIdx]/np.sqrt(len(example//2))
					hasConverged = False 
			if hasConverged: # The algorithm has converged
				break

	def getErr(self, examples, labels):
		successCount = 0

		for exampleIdx, example in enumerate(examples):
			if (self.label(example) == labels[exampleIdx]):
				successCount += 1
		return (1 - successCount / len(examples))

	def output(self, example):
		return np.dot(example, self.weights)

	def label(self, example):
		if self.output(example) > 0: return 1 # this is a fancy way to do a dot product. Note we therefore require both have length N
		else: return -1



