import numpy as np
import math

# Generate a random 2D weight vector P times, and run it through all possible 2D binary inputs

P = 10

binIn = [np.array([-1, 1]),np.array([1, -1]),np.array([-1, -1]), np.array([1, 1])]

weightvectors = [2 * np.random.random((P, 2)) - 1, 2 * np.random.random((P, 2)) - 1]
#weightvectors = [[[-1, 1]],[[1, -1]]]


def relu(X):
   return np.maximum(0,X)

average = 0

for p in range(len(weightvectors[0])):
	for i in range(len(binIn) ):
		print([relu(np.dot(weightvectors[j][p], binIn[i])) for j in range(2)])
		print("teacher: {}".format(-binIn[i][0] * binIn[i][1]))
		print("agreement: {}   \n".format(-binIn[i][0] * binIn[i][1] * np.sum([relu(np.dot(weightvectors[j][p], binIn[i])) for j in range(2)])))
		average += np.sign(-binIn[i][0] * binIn[i][1] * sum([relu(np.dot(weightvectors[j][p], binIn[i])) for j in range(2)]))

print(average)
