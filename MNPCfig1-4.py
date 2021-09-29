import numpy as np
import matplotlib.pyplot as plt
import random
from binaryClassifier import BinaryClassifier
from MNPC import *

# In this file I hope to realize the number ranking problem with a naively and simply implemented
# version of the MNPC, that is, one with unrealistically simplified NPC parameters. 
# The purpose of this simplification is to make sure that each of the M=N NPCs simply outputs 
# a single component of the N-dimensional inputs, such that training the readout weights amounts 
# to just training a perceptron (in theory). This is the theory I want to check. 

#Generate dataset:
def sampleOneExample(N: int):
	# N is twice the length of the binary (-1, 1) encoded number
	return np.array([random.randrange(2) * 2 - 1 for _ in range(N)])

def getTrainingSet(N: int, p: int):
	trainingSet = np.ndarray(shape = (p, N), dtype = int)
	for exampleIdx, _ in enumerate(trainingSet):
		trainingSet[exampleIdx] = sampleOneExample(N)
	return trainingSet

def targetLabel(input):
	inputA = ''
	for digit in input[:len(input)//2]:
		if digit == 1:
			inputA = inputA + '1'
		else:
			inputA = inputA + '0'

	inputB = ''
	for digit in input[len(input)//2:]:
		if digit == 1:
			inputB = inputB + '1'
		else:
			inputB = inputB + '0'

	if int(inputA, 2) > int(inputB, 2): return 1
	else: return -1 

def getGenErr(classifier: BinaryClassifier):
	# I haven't figured out how to get an exact value for the gen err, so here I do it numerically:
	numIter = 100
	testSet = getTrainingSet(classifier.N, numIter) # divided by 2 because the function expects number length, not N
	return classifier.getErr(testSet, np.array([targetLabel(testSet[i]) for i in range(len(testSet))]))


def AnnealedGenErr(alpha: float):
	x = np.linspace(0.001, 0.999, 999)
	f = lambda epsilon: 0.5*np.log((np.sin(np.pi * epsilon)**2)) + alpha * np.log(1 - epsilon)
	fs = f(x)
	return x[np.argmax(fs)]


def runRankingExperiment(numSimulations: int = 1, N: int = 5):
	pList = [i * 10 for i in range(21)]
	generalizationErrors = np.ndarray(shape = (len(pList), numSimulations))
	trainingErrors = []
	for simIdx in range(numSimulations):
		trainingSet = getTrainingSet(N, max(pList))
		trainingLabels = np.array([targetLabel(trainingSet[i]) for i in range(len(trainingSet))])
		mnpc = BinMNPC(N * 3 , N)

		# We train in intervals, therefore we have len(pList) - 1 intervals
		for pIdx, p in enumerate(pList):
			mnpc.train(trainingSet[0:p], trainingLabels)
			generalizationErrors[pIdx, simIdx] = getGenErr(mnpc)
			

	# We trained in intervals, therefore we leave out 1 of the pList in both x- and y-axes 
	plt.plot(pList, [np.mean(generalizationErrors[i, :]) for i in range(len(pList))], label = "Generalization")
	#Obtain generalization error through annealed approximation (see book) and plot
	AnnealedErrorList = [AnnealedGenErr(p / (N)) for p in pList]
	plt.plot(pList, AnnealedErrorList, label = "AA")
	plt.legend()
	plt.show()

runRankingExperiment()