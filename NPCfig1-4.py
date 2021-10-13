import numpy as np
import matplotlib.pyplot as plt
import random
from binaryClassifier import BinaryClassifier
from perceptron import BinPerceptron
from NPC import BinNPC

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

def getGenErr_with_teacher(teacher: BinaryClassifier, classifier: BinaryClassifier):
	# I haven't figured out how to get an exact value for the gen err, so here I do it numerically:
	numIter = 100
	testSet = getTrainingSet(classifier.N, numIter) # divided by 2 because the function expects number length, not N
	return classifier.getErr(testSet, np.array([teacher.label(testSet[i]) for i in range(len(testSet))]))



def AnnealedGenErr(alpha: float):
	x = np.linspace(0.001, 0.999, 999)
	f = lambda epsilon: 0.5*np.log((np.sin(np.pi * epsilon)**2)) + alpha * np.log(1 - epsilon)
	fs = f(x)
	return x[np.argmax(fs)]


def runRankingExperiment(numSimulations: int = 5, N: int = 20):
	pList = [i * 10 for i in range(40)]
	generalizationErrors = np.ndarray(shape = (len(pList), numSimulations))
	trainingErrors = []
	for simIdx in range(numSimulations):
		trainingSet = getTrainingSet(N, max(pList))
		trainingLabels = np.array([targetLabel(trainingSet[i]) for i in range(len(trainingSet))])
		npc = BinNPC(N)

		# We train in intervals, therefore we have len(pList) - 1 intervals
		for pIdx, p in enumerate(pList):
			npc.train(trainingSet[0:p], trainingLabels)
			generalizationErrors[pIdx, simIdx] = getGenErr(npc)


	# We now obtain another curve which shows the typical learning behavior of an NPC in a student-teacher scenario:
	generalizationErrors_NPCteacher = np.ndarray(shape = (len(pList), numSimulations))
	trainingErrors_NPCteacher = []
	for simIdx in range(numSimulations):
		teacher = BinNPC(N)
		trainingSet_NPCteacher = getTrainingSet(N, max(pList))
		trainingLabels_NPCteacher = np.array([teacher.label(trainingSet_NPCteacher[i]) for i in range(len(trainingSet_NPCteacher))])
		npc_NPCteacher = BinNPC(N)

		# We train in intervals, therefore we have len(pList) - 1 intervals
		for pIdx, p in enumerate(pList):
			npc_NPCteacher.train(trainingSet_NPCteacher[0:p], trainingLabels_NPCteacher)
			generalizationErrors_NPCteacher[pIdx, simIdx] = getGenErr_with_teacher(teacher, npc_NPCteacher)

			

	# We trained in intervals, therefore we leave out 1 of the pList in both x- and y-axes 
	plt.plot(pList, [np.mean(generalizationErrors[i, :]) for i in range(len(pList))], label = "Generalization")
	# Now we plot the behavior for a student and teacher NPC, perhaps this is less disappointing than the performance on number ranking:
	plt.plot(pList, [np.mean(generalizationErrors_NPCteacher[i, :]) for i in range(len(pList))], label = "Generalization with NPC teacher")
	#Obtain generalization error through annealed approximation (see book) and plot
	AnnealedErrorList = [AnnealedGenErr(p / (N)) for p in pList]
	plt.plot(pList, AnnealedErrorList, label = "AA for N-perceptron")

	plt.legend()
	plt.show()

runRankingExperiment()