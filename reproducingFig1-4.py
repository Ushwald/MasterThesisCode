# In this script I seek to reproduce figure 1.4 from the book by Engel and Van den Broeck
# This is a plot of learning curves (simulationally, Annealed Approximation, Gardner/replica method analysis)
# I seek to produce the simulated results
# The procedure will apply noiseless Gibbs training to a scenario where two 10-digit binary numbers
# are encoded in a total of N=20 inputs. The binary output then encodes whether the prior or latter number
# should be greater. This is therefore deviant from a standard student-teacher scenario, where both student and teacher
# are networks of similar/identical architecture. 

# Steps will be: 
# Produce a perceptron output function
# Produce an training update criterion and function
# Keep track of training accuracy
# Plot after N runs

import numpy as np
import random
import matplotlib.pyplot as plt

def perceptronOutput(weights, inputs):
	weightsA = np.array(weights)
	inputsA = np.array(inputs)
	return weights @ inputs # this is a fancy way to do a dot product. Note we therefore require both have length N
	

def perceptronLabel(weights, inputs):
	if perceptronOutput(weights, inputs) > 0: return 1 # this is a fancy way to do a dot product. Note we therefore require both have length N
	else: return -1


def sampleOneExample(numberLength: int):
	# Numberlength is the length in binary (-1, 1) encoded number
	return np.array([random.randrange(2) * 2 - 1 for _ in range(numberLength * 2)])

def targetLabel(input, numberLength):
	inputA = ''
	for digit in input[:numberLength]:
		if digit == 1:
			inputA = inputA + '1'
		else:
			inputA = inputA + '0'

	inputB = ''
	for digit in input[numberLength:]:
		if digit == 1:
			inputB = inputB + '1'
		else:
			inputB = inputB + '0'

	if int(inputA, 2) > int(inputB, 2): return 1
	else: return -1 

def trainStep(weights, inputs, numberLength):
	# We apply the perceptron training algorithm, as does the book
	if perceptronLabel(weights, inputs) * targetLabel(inputs, numberLength) <= 0:
		# Update the weight vector:
		weights = weights + inputs * targetLabel(inputs, numberLength)/np.sqrt(numberLength)
	return weights

def getGenErr(weights, numberLength):
	# I haven't figured out how to get an exact value for the gen err, so here I do it numerically:
	numIter = 100
	successCount = 0
	for idx in range(numIter):
		example = sampleOneExample(numberLength)
		if (perceptronLabel(weights, example) == targetLabel(example, numberLength)):
			successCount += 1
	return (1 - successCount / numIter)

def trainOnce(p: int, numberLength: int = 10):
	# Initialize a perceptron (its weights)
	# Then run it through many training iterations with
	# then return the generalization error 
	weights = np.random.rand(numberLength * 2) * 2 - 1 # the book wasn't clear on how the weights were initialized, so I chose this.
	weights = weights / np.linalg.norm(weights) * np.sqrt(numberLength * 2) # To ensure J^2 = N = 20
	for idx in range(p):
		weights = trainStep(weights, sampleOneExample(numberLength), numberLength)
	return getGenErr(weights, numberLength)


def AnnealedGenErr(alpha: float):
	x = np.linspace(0.001, 0.999, 999)
	f = lambda epsilon: 0.5*np.log((np.sin(np.pi * epsilon)**2)) + alpha * np.log(1 - epsilon)
	fs = f(x)
	return x[np.argmax(fs)]

def runRankingExperiment(numSimulations: int = 10, numberLength: int = 10):
	averageGeneralizationErrors = []
	pList = [i * 10 for i in range(20)]
	for p in pList:
		generalizationErrors = []
		for simIdx in range(numSimulations):
			generalizationErrors.append(trainOnce(p, numberLength))
		averageGeneralizationErrors.append(sum(generalizationErrors)/len(generalizationErrors))

	plt.plot(pList, averageGeneralizationErrors)

	#Obtain generalization error through annealed approximation (see book) and plot
	AnnealedErrorList = [AnnealedGenErr(p / (numberLength * 2)) for p in pList]
	plt.plot(pList, AnnealedErrorList)
	plt.show()

runRankingExperiment()

