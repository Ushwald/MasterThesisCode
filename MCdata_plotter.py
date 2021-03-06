
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

plt.style.use('seaborn-whitegrid')

'''
This file plots the experimental simulation data and the analytic results of the generalization error for a 
binary-input N=2 MP classifier in the T->infty limit, with a XOR gate teacher.
'''

#Load the data generated by mathematica
analyticdata = np.loadtxt('data/N2XOR_AA_GenErr.txt').T

fig = plt.figure()

palette = sns.color_palette()

plt.xlabel('$α = β P / N$', fontsize = 16, )
plt.ylabel('Generalization Error $\epsilon_g$', fontsize = 16)
plt.xlim(0,6)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 10)


plt.plot(analyticdata[0], analyticdata[1], color = palette[0], linewidth = 3, label = 'Annealed Approximation')
titles = ['T1000_bigrun', 'T1_bigrun']
for i, title in enumerate(titles):

	with open('data/{}.npy'.format(title), 'rb') as f:
		experiment_metadata = pickle.load(f)
		
	print(experiment_metadata[1:])
	data = experiment_metadata[0]
	alphas = experiment_metadata[1]
	discard = experiment_metadata[6]
	T = int(1/experiment_metadata[2])

	runsData = np.average(data[:, :, discard:], axis = 2)
	plottableData = np.average(runsData, axis = 1)
	stderrs = np.std(runsData, axis = 1)

	print(len(alphas))
	plt.errorbar(x = alphas, y = plottableData, yerr = stderrs, color = palette[i + 1], linewidth = 2, label = 'MC at T = {}'.format(T), linestyle = 'dashed', mew = 2, fmt = '.k', elinewidth = len(titles) - i)

plt.legend()

plt.show()




