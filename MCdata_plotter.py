import numpy as np
import matplotlib.pyplot as plt

#load sim data:
with open('data/MCdata.npy', 'rb') as f:
	data = np.load(f)


discard = 100
alphas = [i+1 for i in range(10)]
plottableData = np.array([np.average(data[a, :, discard:])for a, _ in enumerate(alphas)])

#load mathematica data:
# Skip for now

fig = plt.figure()
plt.plot(alphas, plottableData, color = 'blue')
plt.xlabel('α')
plt.ylabel('Generalization Error')
plt.show()




