import numpy as np

array = [1,5,123,5,1123]

n = np.reshape(array, (len(array), 1))

print(n)

a = np.reshape(n, len(n))

print(a)
