
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

plt.style.use('seaborn-whitegrid')

#load sim data:
#with open('data/MCdata_testlowT.npy', 'rb') as f:
#	data = np.load(f)
#with open('data/MCdata_alpha1through10.npy', 'rb') as f:
#	datahighalpha = np.load(f)
#with open('data/MCdata_lowalpha.npy', 'rb') as f:
#	datalowalpha  = np.load(f)

#Load the data generated by mathematica
analyticdata = np.loadtxt('data/N2XOR_AA_GenErr.txt').T

fig = plt.figure()

palette = sns.color_palette()

plt.xlabel('α', fontsize = 16)
plt.ylabel('Generalization Error', fontsize = 16)
plt.xlim(0,6)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 10)

#titles = ['T1000_bigrun', 'T100_bigrun', 'T10_bigrun', 'T1_bigrun']
titles = ['T1000_bigrun', 'T1_bigrun']
for i, title in enumerate(titles):

	with open('data/{}.npy'.format(title), 'rb') as f:
		experiment_metadata = pickle.load(f)
		#GenErrArray, alphas, beta, runs, MCS, step_scale, discard
	data = experiment_metadata[0]
	alphas = experiment_metadata[1]
	discard = experiment_metadata[6]
	T = int(1/experiment_metadata[2])

	runsData = np.average(data[:, :, discard:], axis = 2)
	plottableData = np.average(runsData, axis = 1)
	stderrs = np.std(runsData, axis = 1)

	print(len(alphas))
	plt.errorbar(x = alphas, y = plottableData, yerr = stderrs, color = palette[i], linewidth = 2, label = 'MC at T = {}'.format(T), linestyle = 'dashed', mew = 2, fmt = '.k', elinewidth = len(titles) - i)
	#for r in runsData.T:
	#	plt.plot(alphas, r)
	#plt.plot(alphas, plottableData, color = palette[i], linewidth = 2, label = 'MC at T = {}'.format(T), mew = 2, marker = 'o', markersize = 4)
	#plt.plot(alphas, np.add(plottableData, 0.5 * stderrs), color = palette[i], linewidth = 1, linestyle = 'dashed')
	#plt.plot(alphas, np.add(plottableData, -0.5 * stderrs), color = palette[i], linewidth = 1, linestyle = 'dashed')



plt.plot(analyticdata[0], analyticdata[1], color = palette[len(titles)], linewidth = 2, label = 'Annealed Approximation')

plt.legend()
#plt.legend(['Annealed Approximation', 'MC at T = 1000 ', 'MC at T = 100', 'MC at T = 10', 'MC at T = 1'])

plt.show()




