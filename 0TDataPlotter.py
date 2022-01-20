import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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
plt.xlim(0,10)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 10)

titles = ['T0_experiment']

for i, title in enumerate(titles):

	with open('data/{}.npy'.format(title), 'rb') as f:
		experiment_metadata = pickle.load(f)
		#(GenErrArray, Plist, runs, MCS, step_scale, discard)
	data = experiment_metadata[0]
	Plist = experiment_metadata[1]
	discard = experiment_metadata[5]

	runsData = np.average(data[:, :, discard:], axis = 2)
	plottableData = np.average(runsData, axis = 1)
	stderrs = np.std(runsData, axis = 1)

	plt.errorbar(x = Plist, y = plottableData, yerr = stderrs, color = palette[i], linewidth = 2)

plt.plot(analyticdata[0], analyticdata[1], color = palette[len(titles)], linewidth = 2)


plt.legend(['T0'])

plt.show()




