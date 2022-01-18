import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load sim data:
with open('data/MCdata.npy', 'rb') as f:
	data = np.load(f)
#with open('data/MCdata_alpha1through10.npy', 'rb') as f:
#	datahighalpha = np.load(f)
#with open('data/MCdata_lowalpha.npy', 'rb') as f:
#	datalowalpha  = np.load(f)

#data = np.concatenate((datalowalpha, datahighalpha))
analyticdata = np.loadtxt('data/N2XOR_AA_GenErr.txt').T

discard = 100
alphas = [(i+1) / 10 for i in range(9)] + [(i + 1) for i in range(10)]
#plottableData = np.array([np.average(data[a, :, discard:]) for a, _ in enumerate(alphas)])
runsData = np.average(data[:, :, discard:], axis = 2)
plottableData = np.average(runsData, axis = 1)
stderrs = np.std(runsData, axis = 1)

#load mathematica data:
# Skip for now

fig = plt.figure()
plt.errorbar(x = alphas, y = plottableData, yerr = stderrs, color = 'blue')
plt.xlabel('α')
plt.ylabel('Generalization Error')
plt.plot(analyticdata[0], analyticdata[1], color = 'green')

palette = sns.color_palette()

#for run in range(data.shape[1]):
	#plt.plot(alphas, [np.average(data[a, run, discard:]) for a, _ in enumerate(alphas)], color = palette[run])

plt.show()




