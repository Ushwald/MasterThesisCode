
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

plt.style.use('seaborn-whitegrid')


fig = plt.figure()

palette = sns.color_palette()

plt.xlabel('$α = β P / N$', fontsize = 16, )
plt.ylabel('Interaction $\omega$', fontsize = 16)
plt.xlim(0,6)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 10)

titles = ['T1000_bigrun_with_omegas']
for i, title in enumerate(titles):

	with open('data/{}.npy'.format(title), 'rb') as f:
		experiment_metadata = pickle.load(f)

	print(experiment_metadata[1:])
	data = experiment_metadata[0]
	alphas = experiment_metadata[1]
	discard = experiment_metadata[6]
	T = int(1/experiment_metadata[2])
	omegas = experiment_metadata[7]

	runsData = np.average(omegas[:, :, discard:], axis = 2)
	plottableData = np.average(runsData, axis = 1)
	stderrs = np.std(runsData, axis = 1)

	print(len(alphas))
	plt.errorbar(x = alphas, y = plottableData, yerr = stderrs, color = palette[i + 1], linewidth = 2, label = 'MC at T = {}'.format(T), linestyle = 'dashed', mew = 2, fmt = '.k', elinewidth = len(titles) - i)
	

plt.legend()

plt.show()




