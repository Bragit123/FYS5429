import numpy as np
from scipy.signal import convolve2d, correlate
import matplotlib.pyplot as plt

image = np.array([
    [
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ], [
        [2, 2, 2, 0],
        [2, 0, 0, 0],
        [2, 2, 0, 0],
        [2, 0, 0, 0]
    ], [
        [3, 3, 3, 0],
        [3, 0, 0, 0],
        [3, 3, 0, 0],
        [3, 0, 0, 0]
    ]
])

kernel1 = np.array([
    [
        [1, 1],
        [0, 0]
    ], [
        [1, 1],
        [0, 0]
    ], [
        [1, 1],
        [0, 0]
    ]
])

kernel2 = np.array([
    [
        [1, 0],
        [1, 0]
    ], [
        [1, 0],
        [-1, 0]
    ], [
        [1, 0],
        [-1, 0]
    ]
])

bias = np.array([
    [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]
    ], [
        [1, -1, 1],
        [-2, 2, 2],
        [3, -3, 3]
    ],
])

# image = image[np.newaxis,:,:]
kernel = np.array([kernel1, kernel2])

image_shape = np.shape(image)
kernel_shape = np.shape(kernel)
output_shape = (kernel_shape[0], image_shape[1]-kernel_shape[2]+1, image_shape[2]-kernel_shape[3]+1)
output = np.zeros(output_shape)
for i in range(kernel_shape[0]):
    for c in range(image_shape[0]):
        corr = correlate(image[c,:,:], kernel[i,c,:,:], "valid") + bias[i,:,:]
        output[i,:,:] = np.sum(corr, axis=1)

def RELU(X):
    return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))

output = RELU(output)

print(output)

plt.imshow(image)
plt.savefig("F.pdf")