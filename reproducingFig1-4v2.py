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

def trainPerceptron(trainingSet):
	# Initialize a perceptron (its weights)
	# Then run it through many training iterations with
	# then return the generalization error 
	weights = np.random.rand(trainingSet.shape[1]) * 2 - 1 # the book wasn't clear on how the weights were initialized, so I chose this.
	for example in trainingSet:
		# We apply the perceptron training algorithm, as does the book
		if np.dot(weights, example) * targetLabel(example) <= 0:
			# Update the weight vector:
			weights = weights + example * targetLabel(example)/np.sqrt(len(example//2))
	return getGenErr(weights)

def AnnealedGenErr(alpha: float):
	x = np.linspace(0.001, 0.999, 999)
	f = lambda epsilon: 0.5*np.log((np.sin(np.pi * epsilon)**2)) + alpha * np.log(1 - epsilon)
	fs = f(x)
	return x[np.argmax(fs)]


def runRankingExperiment(numSimulations: int = 100, numberLength: int = 10):
	averageGeneralizationErrors = []
	pList = [i * 10 for i in range(20)]
	for p in pList:
		generalizationErrors = []
		for simIdx in range(numSimulations):
			generalizationErrors.append(trainPerceptron(getTrainingSet(numberLength, p * 2)))
		averageGeneralizationErrors.append(sum(generalizationErrors)/len(generalizationErrors))

	plt.plot(pList, averageGeneralizationErrors)

	#Obtain generalization error through annealed approximation (see book) and plot
	AnnealedErrorList = [AnnealedGenErr(p / (numberLength * 2)) for p in pList]
	plt.plot(pList, AnnealedErrorList)
	plt.show()

runRankingExperiment()


trainPerceptron(getTrainingSet(10, 3))


