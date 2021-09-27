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

def MNPClayer(omega_i, inputs):
	outputs = []
	for NPC, _ in enumerate(omega_i):
		output = 0
		for binaryInputA in inputs:
			for binaryInputB in inputs:
				output +=  omega_i[NPC][binaryInputA * len(inputs) + binaryInputB] * binaryInputA * binaryInputB
		outputs.append(output)
	return np.array(outputs)

def MNPCOutput(omega_i, weights, inputs):
	return np.dot(np.array(MNPClayer(omega_i, inputs)), np.array(weights))

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

def getErr(weights, exampleSet):
	successCount = 0
	for exampleIdx, _ in enumerate(exampleSet):
		if (perceptronLabel(weights, exampleSet[exampleIdx]) == targetLabel(exampleSet[exampleIdx])):
			successCount += 1
	return (1 - successCount / len(exampleSet))

def getMNPCErr(omega_i, weights, exampleSet):
	successCount = 0
	for exampleIdx, _ in enumerate(exampleSet):
		label = 1 if MNPCOutput(omega_i, weights, exampleSet[exampleIdx]) > 0 else -1
		if (label == targetLabel(exampleSet[exampleIdx])):
			successCount += 1
	return (1 - successCount / len(exampleSet))


def getGenErr(weights):
	# I haven't figured out how to get an exact value for the gen err, so here I do it numerically:
	numIter = 100
	testSet = getTrainingSet(len(weights)//2, numIter)
	return getErr(weights, testSet)

def getMNPCGenErr(omega_i, weights):
	numIter = 100
	testSet = getTrainingSet(len(weights)//2, numIter)
	return getMNPCErr(omega_i, weights, testSet)

def trainPerceptron(startingWeights, trainingSet, untilConvergence = True): # Last bool indicates whether to converge
	weights = startingWeights

	while True:
		hasConverged = True #  If we insist on convergence, we may set this to false
		for example in trainingSet:
			# We apply the perceptron training algorithm, as does the book
			if np.dot(weights, example) * targetLabel(example) <= 0:
				# Update the weight vector:
				weights = weights + example * targetLabel(example)/np.sqrt(len(example//2))
				if untilConvergence: hasConverged = False 
		if hasConverged: # The algorithm has converged
			break
	return weights

def trainMNPC(omega_i, startingWeights, trainingSet, untilConvergence = True): # Last bool indicates whether to converge
	# We use the perceptron training algorithm
	weights = startingWeights

	while True:
		hasConverged = True #  If we insist on convergence, we may set this to false
		for example in trainingSet:
			# We apply the perceptron training algorithm, as does the book
			if MNPCOutput(omega_i, weights, example) * targetLabel(example) <= 0:
				# Update the weight vector: (not sure if I normalize this right)
				weights = weights + MNPClayer(omega_i, example) * targetLabel(example)/np.sqrt(M)
				if untilConvergence: hasConverged = False 
		if hasConverged: # The algorithm has converged
			break
	print("At least one did converge, for {} examples".format(trainingSet.shape[0]))
	return weights

def AnnealedGenErr(alpha: float):
	x = np.linspace(0.001, 0.999, 999)
	f = lambda epsilon: 0.5*np.log((np.sin(np.pi * epsilon)**2)) + alpha * np.log(1 - epsilon)
	fs = f(x)
	return x[np.argmax(fs)]


def runPerceptronRankingExperiment(numSimulations: int = 100, numberLength: int = 10):
	pList = [i * 10 for i in range(21)]
	generalizationErrors = np.ndarray(shape = (len(pList), numSimulations))
	trainingErrors = []
	print(generalizationErrors.shape)
	for simIdx in range(numSimulations):
		trainingSet = getTrainingSet(numberLength, max(pList))
		weights = np.random.rand(trainingSet.shape[1]) * 2 - 1 # initialize perceptron

		# We train in intervals, therefore we have len(pList) - 1 intervals
		for pIdx, p in enumerate(pList):
			weights = trainPerceptron(weights, trainingSet[0:p])
			generalizationErrors[pIdx, simIdx] = getGenErr(weights)
			

	# We trained in intervals, therefore we leave out 1 of the pList in both x- and y-axes 
	plt.plot(pList, [np.mean(generalizationErrors[i, :]) for i in range(len(pList))], label = "Generalization")
	#Obtain generalization error through annealed approximation (see book) and plot
	AnnealedErrorList = [AnnealedGenErr(p / (numberLength * 2)) for p in pList]
	plt.plot(pList, AnnealedErrorList, label = "AA")
	plt.legend()
	plt.show()

def runMNPCRankingExperiment(M: int = 10, numSimulations: int = 10, numberLength: int = 3): # M is the number of NPC in the layer
	# Same setup as in the perceptron case, except now we train some random instantiation of an MNPC for binary inputs
	# and plot each of their performances
	pList = [i * 10 for i in range(21)]
	generalizationErrors = np.ndarray(shape = (len(pList), numSimulations))
	trainingErrors = []
	print(generalizationErrors.shape)
	for simIdx in range(numSimulations):
		trainingSet = getTrainingSet(numberLength, max(pList))

		# omega_i being the NPC parameters, i upto N^2
		omega_i = [np.random.rand(trainingSet.shape[1] ** 2) for _ in range(M)]	# To be later made to conform to empirical distributions
		weights = np.random.rand(M) * 2 - 1 # initialize MNPC weights

		# We train in intervals, therefore we have len(pList) - 1 intervals
		for pIdx, p in enumerate(pList):
			weights = trainMNPC(omega_i, weights, trainingSet[0:p])
			generalizationErrors[pIdx, simIdx] = getMNPCGenErr(omega_i, weights)
			

	# We trained in intervals, therefore we leave out 1 of the pList in both x- and y-axes 
	plt.plot(pList, [np.mean(generalizationErrors[i, :]) for i in range(len(pList))], label = "Generalization")
	


runMNPCRankingExperiment(3)



