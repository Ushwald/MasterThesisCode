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
		self.omega_i = np.random.rand(M, N, N)
		self.perceptron = BinPerceptron(M) # The perceptron takes inputs from each of the M NPC's

	def label(self, example):
		print(example.T * example)





# Temporary debugging stuff:

def sampleOneExample(numberLength: int):
	# Numberlength is the length in binary (-1, 1) encoded number
	return np.array([random.randrange(2) * 2 - 1 for _ in range(numberLength * 2)])

def getTrainingSet(numberLength: int, p: int):
	trainingSet = np.ndarray(shape = (p, 2 * numberLength), dtype = int)
	for exampleIdx, _ in enumerate(trainingSet):
		trainingSet[exampleIdx] = sampleOneExample(numberLength)
	return trainingSet


N = 8
M = 4
classifier = BinMNPC(M, N)
example = sampleOneExample(N //2)
print(example)
print(classifier.label(example))
