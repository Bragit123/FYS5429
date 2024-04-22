import numpy as np
from jax import vmap
from funcs import softmax

X = np.array([[1,2,1e10],[1,2,1e10]])

print(softmax(X))