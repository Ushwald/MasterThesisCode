"""
This file will contain the numerical integration of version space.

This will proceed as follows:
- A given teacher (W and omega) is fixed
- A grid is composed of possible values of student W and omega.
	- This will be restricted to a hypersphere in N^2 dimensions
	- Only grids which contain points on the hypersphere are selected (subject to later change)
	- This selection takes place by seeing if the inner diagonal of a box is inside the radius, and the 
		outer diagonal is outside the radius
- Each grid cell corresponds to a student configuration, and for each of these, some number of examples are 
	sampled from a Gaussian distribution, and the fraction of samples which is classified equally by student
	and teacher is recorded, then assigned to that grid cell. 
- The sum of each grid cell agreement fraction is stored, and gives the version space for that teacher


"""

import numpy as np
from itertools import product

# Setting the relevant parameters:
N = 3 				# dimensions of the inputs
p = 1 				# Number of examples in one input
normalization = N 	# Similar to the perceptron case, normalize the radius to equal N (not that it matters)
gridRes = 4		# The number of grid cells to be fitted in one radius
gridStep = normalization / gridRes

teacher =  np.zeros(shape = (N, N))	# Diagonals are the W, off-diagonals are the omega
for i in range(N):
	teacher[i, i] = 1.0


# We begin by producing the grid as an array of points, each corresponding to the cell's center
tmp = [i for i in range(gridRes)]
possibleCoords = [-i for i in range(1, gridRes)]
possibleCoords.reverse()
possibleCoords += tmp
print(int(N**2/2 + N/2))
print(len(possibleCoords))

def getPos(coord):
	return np.array([c * gridStep for c in coord])

def containsRadius(coord, r):
	# we say a cell contains the radius when its center's positions's norm is
	# between r-gridstep/2 and r+gridstep/2:
	posNormDiff = np.linalg.norm(getPos(coord)) - r
	return posNormDiff > -gridStep/2 and posNormDiff < gridStep/2


# The following generates coordinate indices for each of the grid cells. 
# This could really explode quickly if we don't watch out with the dimensionality of the problem
grid = []
for coord in product(possibleCoords, repeat = int(N**2/2 + N/2)):
	if containsRadius(coord, normalization):
		grid.append(getPos(coord))	# Store the centers of grid cells on the hypersphere
		
print(len(grid))
