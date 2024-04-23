import numpy as np
from jax import vmap
from funcs import softmax


kernel_size = (2,2,2,1)



for i in range(0, input_height - kernel_height, stride): #can change 1 with stride possibly
            for j in range(0, input_width - kernel_width, stride):
                for d in range(num_kernels):
                    output[:, i, j, d] = np.sum(input[:, i : i + self.kernel_height, j : j + self.kernel_width, :] * self.kernels[d, :, :, :], axis=3)



X = np.array([[1,2,1e10],[1,2,1e10]])

print(softmax(X))