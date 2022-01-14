import random
import numpy as np
import matplotlib.pyplot as plt
import math

# This file serves to obtain by monte carlo simulation the generalization error for nonzero temperature
# and thereby give empirical data to compare with the AA obtained result. 

"""Steps involved:

Repeat the following a large number of times, and store the average Gen Err for each run and step:
-	Randomly sample P binary inputs (-1, 1). Let P be large, so that P beta is alpha.
- 	Randomly sample a (W1, W2, Omega) vector from a sphere (or include threshold for hypersphere)
- 	For each monte carlo step:
- 	Randomly add Gaussian noise to the parameters, then renormalize (such that acceptance rate is around 0.5)
- 	Compute the difference in energy DeltaE associated with the change in parameters
-	Accept the change with probability min(1, exp(-beta DeltaE))
-	Regardless of whether it is accepted, compute the generalization error, by simply computing the average MSE for each of the 4 possible inputs

-	Discard results from the first 1000 or so MCS, and egin keeping track once the Energy is stable. 
-	For each of the 'interesting' steps, take the average generalization error and the standard deviation over all the runs
	
"""

inputs = [[1, 1], [1, -1], [-1, 1], [-1, -1]]

def label(W1, W2, Omega, example):
	return np.sign(W1 * example[0] + W2 * example[1] + Omega * example[0] * example[1])

def GetError(W1, W2, Omega,example):
	return (label(W1, W2, Omega, example) + example[0] * example[1])**2/2

def getGenError(W1, W2, Omega):
	return sum([GetError(W1, W2, Omega, inp) for inp in inputs])/len(inputs)

def RandInputs(P):
	return [[random.randint(0, 1) * 2 - 1, random.randint(0, 1) * 2 - 1] for _ in range(P)]

def RandConfig(nParams = 3):
	config = np.random.random((1, 3)) * 2 - 1
	return config[0] / np.linalg.norm(config) * np.sqrt(2) # N = 2

def RunMCSNoThreshold(beta, alphas, runs, MCS, discard = 100):
	# Randomly initialize the parameters from a sphere
	W1, W2, Omega = RandConfig()

	GenErrArray = np.ndarray((len(alphas), runs, MCS + 1))

	for alphaidx in range(len(alphas)):

		for run in range(runs):
			# Get training data:
			trainInputs = RandInputs(round(alphas[alphaidx] / beta * 2))
			targetOutputs = [-inp[0] * inp[1] for inp in trainInputs]

			step = 0
			acceptcount = 0
			nacceptcount = 0
			while(True):
				# Store the generalization error:
				GenErrArray[alphaidx, run, step] = getGenError(W1, W2, Omega)
				# Calculate energy/training error:
				E = sum([ GetError(W1, W2, Omega, trainInputs[i]) for i in range(len(trainInputs))])
				
				#Make an adjustment to the parameters, and determine whether it is to be accepted:
				newconf =  [W1, W2, Omega].copy()
				for idx in range(len(newconf)):
					newconf[idx] += np.random.normal(scale = 0.3)
					
				newconf = newconf/np.linalg.norm(newconf)
				
				newE = sum([GetError(newconf[0], newconf[1], newconf[2], trainInputs[i]) for i in range(len(trainInputs))])
				
				if np.random.random() < min(1, np.exp(-beta * (newE - E))):
					[W1, W2, Omega] = newconf
					acceptcount += 1
					pass
				else:
					nacceptcount +=1

				if step >= MCS:
					break

				step += 1
			print(acceptcount, nacceptcount)
	print("test")
	fig = plt.figure()
	plottableData = np.array([np.average(GenErrArray[a, :, discard:])for a, _ in enumerate(alphas)])
	with open('data/MCdata.npy', 'wb') as f:
		np.save(f, GenErrArray)

	
	print(plottableData)
	
	plt.plot([GenErrArray[a, 0, :] for a in range(len(alphas))][0])
	plt.show()
		
RunMCSNoThreshold(beta = 0.001, alphas = [i + 1 for i in range(10)], runs = 10, MCS = 500, discard = 100)

