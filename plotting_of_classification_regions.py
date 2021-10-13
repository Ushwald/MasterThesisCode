import matplotlib.pyplot as plt
import numpy as np
from binaryClassifier import *
from NPC import BinNPC


#Let's define a 200x200 search space, of small increments, so that we can see what 
# The classification regions in a 2D example look like. Afterwards we may do 3D.



res = 200
domain = 4.0 # How far along each axis do we go (so: xmax/2 = -xmin/2 and ymax/2 = -ymin/2)

xFigs = 4
yFigs = 4

f, axarr = plt.subplots(xFigs,yFigs)


for figNumX in range(xFigs):
	for figNumY in range(yFigs):
		npc = BinNPC(2)

		xco = []
		yco = []

		for xidx in range(res):
			for yidx in range(res):
				x = (xidx -res/2)/res * domain
				y = (yidx -res/2)/res * domain
				label = npc.label(np.array([x,y]))
				if(label == 1):
					xco.append(x)
					xco.append(x)
					yco.append(y)
					yco.append(y)
				elif (label == -1):
					xco.append(x)
					yco.append(y)
				else:
					print("Invalid label detected: {}, {}".format(x, y))

		xco = np.array(xco)
		yco = np.array(yco)

		H, xedges, yedges = np.histogram2d(xco, yco, bins=res, range=None, normed=None, weights=None, density=None)
		
		axarr[figNumX,figNumY].imshow(H, interpolation='nearest', origin='lower',
		        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

plt.show()