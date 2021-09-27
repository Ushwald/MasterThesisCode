class BinaryClassifier:

	N = None # The input dimension this system was designed for

	def getErr(self, examples, labels) -> float:
		"""Get error percentage on given training set for current configuration of classifier"""
		pass

	def train(self, examples, labels) -> None:
		"""Train the classifier on given examples, until satisfied, using appropriate algorithm"""
		pass

	def label(self, example) -> int:
		"""return either -1 or 1, depending on classifier"""
		pass