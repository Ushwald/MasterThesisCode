from perceptron import BinPerceptron
from binaryClassifier import BinaryClassifier
import numpy as np
import random

class BinNPC:
	"""This version of the NPC takes in symmetric binary inputs (-1 or 1)
	It works simply by implicitly defining an N^2 perceptron and assigning the appropriate N^2 weights"""

	def __init__(self, N: int):
		# initialize a random MNPC with M clusters in the layer
		self.N = N
		# Initialize NPC paremeters however you wish. Actual restrictions on NPC parameters should also go into the training function
		self.W = (np.random.rand(N) - 0.5) * 2
		self.omega_i = (np.random.rand(N, N) - 0.5) * 2 
		for i in range(N):
			self.omega_i[i, i] = 0

	
		self.perceptron = BinPerceptron(N^2) # The perceptron takes inputs from each of the M NPC's
		self.perceptron.weights = np.append(self.W, self.omega_i)

	def transformInput(self, example):
		# Expects -1, 1 input
		# Transform inputs from N-dimensional to N^2-dimensional by adding all 1st order interactions:
		# Return as a tuple: the first being the original input, the recond being the interactions:
		interactions = []
		for i in range(self.N):
			for j in range(self.N):
				if i == j:
					interactions.append(0)
				else:
					interactions.append(example[i] * example[j])
		return np.append(np.array(example), np.array(interactions))


	def train(self, examples, labels):
		# Train the perceptron on the transformed dataset:
		# So first transform the dataset:

		transformedExamples = []
		for _, example in enumerate(examples):
			transformedExamples.append(self.transformInput(example))


		while True:
			hasConverged = True # may be set to false if update was still required
			for exampleIdx, example in enumerate(transformedExamples):
				# We apply the perceptron training algorithm, as does the book
				if self.perceptron.label(example) * labels[exampleIdx] <= 0:
					# Update the weight vector:
					self.perceptron.weights = self.perceptron.weights + example * labels[exampleIdx]/np.sqrt(self.N)
					hasConverged = False 
			if hasConverged: # The algorithm has converged
				break

	def label(self, example):
		return 	self.perceptron.label(self.transformInput(example))


	def getErr(self, examples, labels):
		successCount = 0

		for exampleIdx, example in enumerate(examples):
			if (self.label(example) == labels[exampleIdx]):
				successCount += 1
		return (1 - successCount / len(examples))
