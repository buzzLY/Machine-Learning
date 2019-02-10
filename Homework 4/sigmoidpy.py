import numpy as np
import math
def sigmoid(x):

  	return 1 / (1 + np.exp(-x))



x = np.array([12,3,4])

print (x)
print (sigmoid(x))