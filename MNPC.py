import BinPerceptron
import numpy as np

class BinMNPC:

	def __init__(self, M: int, N: int):
		# initialize a random MNPC with M clusters in the layer
		self.omega_i = np.ndarray(shape = (M, N, N), dtype = float)
		self.weights = Perceptron()