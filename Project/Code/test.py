import numpy as np
from jax import vmap
from funcs import softmax


kernel_size = (2,2,2,1)

input=np.array([[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]])
kernel = np.array([[1,2],[3,4]])

stride = 2

z = np.zeros((3,3))
for i in range(0, 4+1, stride): #can change 1 with stride possibly
    for j in range(0, 4+1, stride):
        print(i)
        z[int(i/stride), int(j/stride)] = np.sum(input[i : i + 2, j : j + 2] * kernel, axis=(0,1))

pad_z = np.zeros((5,5))
pad_z[1:4,1:4] = z

grad_input = np.zeros((6,6))
for i in range(0, 6-2+1, stride): #can change 1 with stride possibly
    for j in range(0, 6-2+1, stride):
                        #z[:, i, j, d] = np.sum(input[:, i : i + self.kernel_height, j : j + self.kernel_width, :] * self.kernels[d, :, :, :], axis=(1,2))[:,0]

        grad_input[i,j] = np.sum(pad_z[int(i/stride) : int(i/stride) + 2, int(j/stride) : int(j/stride) + 2]*kernel, axis=(0,1)) ##PS: add stride in this one
        ##grad_input[i+1,j] = grad_input[i,j]
        #grad_input[i,j+1] = grad_input[i,j]
        #grad_input[i+1,j+1] = grad_input[i,j]

        #for i in range(1, self.input_height - self.kernel_height, self.stride): #can change 1 with stride possibly
         #   for j in range(1, self.input_weight - self.kernel_width, self.stride):
          #      grad_input[:,i,j,:] = grad_input
print(grad_input)
for i in range(0, 6-2+1, stride): #can change 1 with stride possibly
    for j in range(0, 6-2+1, stride):
        grad_input[i+1,j] = grad_input[i,j]
        grad_input[i,j+1] = grad_input[i,j]
        grad_input[i+1,j+1] = grad_input[i,j]

print(z)
print(grad_input)

#X = np.array([[1,2,1e10],[1,2,1e10]])

#print(softmax(X))