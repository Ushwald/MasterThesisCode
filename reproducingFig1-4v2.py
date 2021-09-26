import numpy as np
import matplotlib.pyplot as plt
import random

#Generate dataset:
def sampleOneExample(numberLength: int):
	# Numberlength is the length in binary (-1, 1) encoded number
	return np.array([random.randrange(2) * 2 - 1 for _ in range(numberLength * 2)])

def getTrainingSet(numberLength: int, p: int):
	trainingSet = np.ndarray(shape = (p, 2 * numberLength), dtype = int)
	for exampleIdx, _ in enumerate(trainingSet):
		trainingSet[exampleIdx] = sampleOneExample(numberLength)
	return trainingSet


def perceptronLabel(weights, inputs):
	if np.dot(weights, inputs) > 0: return 1 # this is a fancy way to do a dot product. Note we therefore require both have length N
	else: return -1

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

def getGenErr(weights):
	# I haven't figured out how to get an exact value for the gen err, so here I do it numerically:
	numIter = 100
	successCount = 0
	testSet = getTrainingSet(len(weights)//2, numIter)
	for exampleIdx, _ in enumerate(testSet):
		if (perceptronLabel(weights, testSet[exampleIdx]) == targetLabel(testSet[exampleIdx])):
			successCount += 1
	return (1 - successCount / numIter)

def trainPerceptron(startingWeights, trainingSet):
	weights = startingWeights
	for example in trainingSet:
		# We apply the perceptron training algorithm, as does the book
		if np.dot(startingWeights, example) * targetLabel(example) <= 0:
			# Update the weight vector:
			weights = startingWeights + example * targetLabel(example)/np.sqrt(len(example//2))
	return weights

def AnnealedGenErr(alpha: float):
	x = np.linspace(0.001, 0.999, 999)
	f = lambda epsilon: 0.5*np.log((np.sin(np.pi * epsilon)**2)) + alpha * np.log(1 - epsilon)
	fs = f(x)
	return x[np.argmax(fs)]


def runRankingExperiment(numSimulations: int = 10, numberLength: int = 10):
	pList = [i * 10 for i in range(21)]
	generalizationErrors = np.ndarray(shape = (len(pList), numSimulations))
	print(generalizationErrors.shape)
	for simIdx in range(numSimulations):
		trainingSet = getTrainingSet(numberLength, max(pList))
		weights = np.random.rand(trainingSet.shape[1]) * 2 - 1 # initialize perceptron

		# We train in intervals, therefore we have len(pList) - 1 intervals
		for pIdx in range(len(pList)-1):
			weights = trainPerceptron(weights, trainingSet[pList[pIdx]:pList[pIdx + 1]])
			generalizationErrors[pIdx, simIdx] = getGenErr(weights)

	# We trained in intervals, therefore we leave out 1 of the pList in both x- and y-axes 
	plt.plot(pList[1:], [np.mean(generalizationErrors[i, :]) for i in range(len(pList) - 1)])

	#Obtain generalization error through annealed approximation (see book) and plot
	AnnealedErrorList = [AnnealedGenErr(p / (numberLength * 2)) for p in pList]
	plt.plot(pList, AnnealedErrorList)
	plt.show()

runRankingExperiment()



