import numpy as np
import matplotlib.pyplot as plt
import random
from binaryClassifier import BinaryClassifier
from perceptron import BinPerceptron

#Generate dataset:
def sampleOneExample(numberLength: int):
	# Numberlength is the length in binary (-1, 1) encoded number
	return np.array([random.randrange(2) * 2 - 1 for _ in range(numberLength * 2)])

def getTrainingSet(numberLength: int, p: int):
	trainingSet = np.ndarray(shape = (p, 2 * numberLength), dtype = int)
	for exampleIdx, _ in enumerate(trainingSet):
		trainingSet[exampleIdx] = sampleOneExample(numberLength)
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
	testSet = getTrainingSet(classifier.N//2, numIter) # divided by 2 because the function expects number length, not N
	return classifier.getErr(testSet, np.array([targetLabel(testSet[i]) for i in range(len(testSet))]))


def AnnealedGenErr(alpha: float):
	x = np.linspace(0.001, 0.999, 999)
	f = lambda epsilon: 0.5*np.log((np.sin(np.pi * epsilon)**2)) + alpha * np.log(1 - epsilon)
	fs = f(x)
	return x[np.argmax(fs)]


def runRankingExperiment(numSimulations: int = 100, numberLength: int = 10):
	pList = [i * 10 for i in range(21)]
	generalizationErrors = np.ndarray(shape = (len(pList), numSimulations))
	trainingErrors = []
	for simIdx in range(numSimulations):
		trainingSet = getTrainingSet(numberLength, max(pList))
		trainingLabels = np.array([targetLabel(trainingSet[i]) for i in range(len(trainingSet))])
		ptron = BinPerceptron(numberLength * 2)

		# We train in intervals, therefore we have len(pList) - 1 intervals
		for pIdx, p in enumerate(pList):
			ptron.train(trainingSet[0:p], trainingLabels)
			generalizationErrors[pIdx, simIdx] = getGenErr(ptron)
			

	# We trained in intervals, therefore we leave out 1 of the pList in both x- and y-axes 
	plt.plot(pList, [np.mean(generalizationErrors[i, :]) for i in range(len(pList))], label = "Generalization")
	#Obtain generalization error through annealed approximation (see book) and plot
	AnnealedErrorList = [AnnealedGenErr(p / (numberLength * 2)) for p in pList]
	plt.plot(pList, AnnealedErrorList, label = "AA")
	plt.legend()
	plt.show()

runRankingExperiment()



