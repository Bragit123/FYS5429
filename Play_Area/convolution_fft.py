import numpy as np
from tensorflow.keras import datasets
from numpy.fft import fft2, ifft2, fftn, ifftn, ifftshift
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import time


# image = np.array([
#     [0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 0],
#     [0, 1, 0, 0, 0, 0],
#     [0, 1, 1, 1, 0, 0],
#     [0, 1, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1]
# ])

(X_train, t_train), (X_test, t_test) = datasets.cifar10.load_data()
X_train = np.swapaxes(np.swapaxes(X_train, 1, 3), 2, 3)

image = X_train[0] / 255.0

plt.figure()
plt.imshow(np.swapaxes(np.swapaxes(image, 0, 2), 0, 1))
plt.savefig("F.pdf")

kernel = np.array([
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ],
    [
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ],
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
])

t0 = time.time()
con_image = np.zeros(image.shape)
for i in range(image.shape[0]):
    con_image[i,:,:] = convolve2d(image[i,:,:], kernel[i,:,:], mode="same")

t1 = time.time()
print(f"Convolve time: {t1-t0}")

plt.figure()
plt.imshow(np.swapaxes(np.swapaxes(con_image, 0, 2), 0, 1))
plt.savefig("F_con.pdf")

t0 = time.time()
h_pad = int((image.shape[1]-kernel.shape[1]) / 2)
h_rest = (image.shape[1]-kernel.shape[1]) % 2
w_pad = int((image.shape[2]-kernel.shape[2]) / 2)
w_rest = (image.shape[2]-kernel.shape[2]) % 2
kernel_pad = np.pad(kernel, ((0, 0), (h_pad+h_rest, h_pad), (w_pad+w_rest, w_pad)))
kernel_pad = ifftshift(kernel_pad)

# image_fft = fft2(image)
# kernel_fft = fft2(kernel_pad)
N = image.shape[0] * image.shape[1] * image.shape[2]
image_fft = fftn(image)
kernel_fft = fftn(kernel_pad)

# print(f"image = {image.shape}")
# print(f"kernel = {kernel.shape}")
# print(f"image_fft = {image_fft.shape}")
# print(f"kernel_fft = {kernel_fft.shape}")

im_ker_product = image_fft*kernel_fft
result_fft = ifftn(im_ker_product)
result_fft = np.abs(result_fft) # Me no like complex numbers :'(
t1 = time.time()
print(f"FFT time: {t1-t0}")

# print(f"result_fft = {result_fft.shape}")
# print(result_fft)

print(result_fft)

plt.figure()
plt.imshow(np.swapaxes(np.swapaxes(result_fft, 0, 2), 0, 1))
plt.savefig("F_fft.pdf")