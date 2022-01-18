import numpy as np
import matplotlib.pyplot as plt
import math
import itertools


N = 2
beta = 0.1
runs = 1 #Number of MC experiments to conduct, to average over
alphas = [i+1 for i in range(10)]
steps = 500
discard = 100 #Will have to be tuned later
scenario = 'binary' # or Gaussian
#scenario = gaussian



def generateTrainingData(P):
	if scenario == 'binary':
		return np.array([[1 if np.random.random() > 0.5 else -1 for _ in range(N)]for _ in range(P)])
	elif scenario == 'gaussian':
		return np.random.normal(size = (P, N))

def RunMC(featureSet, labelSet):
	# Generate random config
	# Update, accept, steps times
	# Discards first #discard steps
	pass #Returns errors throughout

def RunExpriment():
	# Run the entire experiment, generate plot
	E_arr = np.ndarray(alpha, runs, steps) # To be later averaged per alpha
	for aidx, alpha in enumerate(alphas):
		[training_features, training_labels] = generateTrainingData(int(alpha / beta * N))
		for run in range(runs):
			# Initialize:
			config = RandConfig()
			for step in range(steps):
				pass
