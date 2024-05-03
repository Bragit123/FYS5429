import numpy as np
from jax import vmap
from funcs import softmax, padding
from scipy.signal import correlate2d, convolve2d



kernel_size = (2,2,2,1)

#input=np.array([[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]])
#kernel = np.array([[1,2],[3,4]])

img=np.random.rand(1,10,10,1)
ker=np.random.rand(1,3,3,1)

ker_h = ker.shape[1]
ker_w = ker.shape[2]

img_h = img.shape[1]
img_w = img.shape[2]
img_d = img.shape[3]

i0 = np.repeat(np.arange(ker_h), ker_h)
i1 = np.repeat(np.arange(img_h), img_h)
j0 = np.tile(np.arange(ker_w), ker_h)
j1 = np.tile(np.arange(img_h), img_w)
i = i0.reshape(-1,1) + i1.reshape(1,-1)
j = j0.reshape(-1,1) + j1.reshape(1,-1)
k = np.repeat(np.arange(img_d), ker_h*ker_w).reshape(-1,1)

#pad_img = padding(img)
#print(i,j)
select_img = img[:,i,j,:].squeeze()
weights = ker.reshape(ker_h*ker_w,-1)
convolve = weights.transpose()@select_img
convolve = convolve.reshape(img.shape)
print(convolve[0,:,:,0])

z = correlate2d(padding(img)[0,:,:,0], ker[0,:,:,0], "valid") 
#z = np.sum(corr, axis=3)
print("z:")
print(z- convolve[0,:,:,0])



#stride = 2

# z = np.zeros((3,3))
# for i in range(0, 4+1, stride): #can change 1 with stride possibly
#     for j in range(0, 4+1, stride):
#         print(i)
#         z[int(i/stride), int(j/stride)] = np.sum(input[i : i + 2, j : j + 2] * kernel, axis=(0,1))

# pad_z = np.zeros((5,5))
# pad_z[1:4,1:4] = z

# grad_input = np.zeros((6,6))
# for i in range(0, 6-2+1, stride): #can change 1 with stride possibly
#     for j in range(0, 6-2+1, stride):
#                         #z[:, i, j, d] = np.sum(input[:, i : i + self.kernel_height, j : j + self.kernel_width, :] * self.kernels[d, :, :, :], axis=(1,2))[:,0]

#         grad_input[i,j] = np.sum(pad_z[int(i/stride) : int(i/stride) + 2, int(j/stride) : int(j/stride) + 2]*kernel, axis=(0,1)) ##PS: add stride in this one
#         ##grad_input[i+1,j] = grad_input[i,j]
#         #grad_input[i,j+1] = grad_input[i,j]
#         grad_input[i+1,j+1] = grad_input[i,j]

#         #for i in range(1, self.input_height - self.kernel_height, self.stride): #can change 1 with stride possibly
#          #   for j in range(1, self.input_weight - self.kernel_width, self.stride):
#           #      grad_input[:,i,j,:] = grad_input
# print(grad_input)
# for i in range(0, 6-2+1, stride): #can change 1 with stride possibly
#     for j in range(0, 6-2+1, stride):
#         grad_input[i+1,j] = grad_input[i,j]
#         grad_input[i,j+1] = grad_input[i,j]
#         grad_input[i+1,j+1] = grad_input[i,j]

# print(z)
# print(grad_input)

#X = np.array([[1,2,1e10],[1,2,1e10]])

#print(softmax(X))