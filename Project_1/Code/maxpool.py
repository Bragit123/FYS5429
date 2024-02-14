import jax.numpy as jnp

class MaxPool:
    def __init__(self, input_size, scale_factor, stride, seed=100):
        self.input_size = input_size
        self.num_inputs, self.input_depth, self.input_height, self.input_width = self.input_size
        self.scale_height, self.scale_width = scale_factor
        self.stride = stride

        self.output_height = int(jnp.floor((self.input_height - self.scale_height) / self.stride) + 1)
        self.output_width = int(jnp.floor((self.input_width - self.scale_width) / self.stride) + 1)
        self.output_size = (self.num_inputs, self.input_depth, self.output_height, self.output_width)
        self.output = jnp.zeros(self.output_size)
    
    def feed_forward(self, input):
        output = jnp.zeros(self.output_size)

        for i in range(self.output_height):
            for j in range(self.output_width):
                h_start = i*self.stride
                h_end = h_start + self.scale_height
                w_start = j*self.stride
                w_end = w_start + self.scale_width
                input_hw = input[:,:, h_start:h_end, w_start:w_end]
                output = output.at[:,:,i,j].set(jnp.max(input_hw, axis=(2,3)))

        return output
    


# import numpy as np
# from scipy.signal import convolve2d, correlate
# import matplotlib.pyplot as plt

# image = np.array([
#     [
#         [1, 1, 1, 0],
#         [1, 0, 0, 0],
#         [1, 1, 0, 0],
#         [1, 0, 0, 0]
#     ], [
#         [2, 2, 2, 0],
#         [2, 0, 0, 0],
#         [2, 2, 0, 0],
#         [2, 0, 0, 0]
#     ], [
#         [3, 3, 3, 0],
#         [3, 0, 0, 0],
#         [3, 3, 0, 0],
#         [3, 0, 0, 0]
#     ]
# ])

# kernel1 = np.array([
#     [
#         [1, 1],
#         [0, 0]
#     ], [
#         [1, 1],
#         [0, 0]
#     ], [
#         [1, 1],
#         [0, 0]
#     ]
# ])

# kernel2 = np.array([
#     [
#         [1, 0],
#         [1, 0]
#     ], [
#         [1, 0],
#         [-1, 0]
#     ], [
#         [1, 0],
#         [-1, 0]
#     ]
# ])

# bias = np.array([
#     [
#         [1, 1, 1],
#         [2, 2, 2],
#         [3, 3, 3]
#     ], [
#         [1, -1, 1],
#         [-2, 2, 2],
#         [3, -3, 3]
#     ],
# ])

# # image = image[np.newaxis,:,:]
# kernel = np.array([kernel1, kernel2])

# image_shape = np.shape(image)
# kernel_shape = np.shape(kernel)
# output_shape = (kernel_shape[0], image_shape[1]-kernel_shape[2]+1, image_shape[2]-kernel_shape[3]+1)
# output = np.zeros(output_shape)
# for i in range(kernel_shape[0]):
#     for c in range(image_shape[0]):
#         corr = correlate(image[c,:,:], kernel[i,c,:,:], "valid") + bias[i,:,:]
#         output[i,:,:] = np.sum(corr, axis=1)

# def RELU(X):
#     return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))

# output = RELU(output)

# print(output)

# plt.imshow(image)
# plt.savefig("F.pdf")