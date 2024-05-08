import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

X = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]
])

K = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0]
])

plt.figure()
plt.imshow(X)
plt.savefig("F.pdf")

X_sp_valid = correlate2d(X, K, mode="valid")
X_sp_same = correlate2d(X, K, mode="same")

plt.figure()
plt.imshow(X_sp_valid)
plt.savefig("sp_valid.pdf")

plt.figure()
plt.imshow(X_sp_same)
plt.savefig("sp_same.pdf")

def correlate_valid(X, K):
    X_h, X_w = X.shape
    K_h, K_w = K.shape
    z_h = X_h - K_h + 1
    z_w = X_w - K_w + 1

    i0 = np.repeat(np.arange(K_h), K_w).reshape((-1, 1))
    i1 = np.repeat(np.arange(z_h), z_w).reshape((1, -1))
    j0 = np.tile(np.arange(K_w), K_h).reshape((-1, 1))
    j1 = np.tile(np.arange(z_w), z_h).reshape((1, -1))

    i = i0 + i1
    j = j0 + j1

    Xp = X[i,j]
    K = np.ravel(K)

    conv_flat = K @ Xp
    conv = conv_flat.reshape((z_h, z_w))

    return conv

def correlate_same(X, K):
    K_h, K_w = K.shape
    pad_top = int(np.ceil((K_h-1)/2))
    pad_bot = int(np.floor((K_h-1)/2))
    pad_left = int(np.ceil((K_w-1)/2))
    pad_right = int(np.floor((K_w-1)/2))

    X = np.pad(X, ((pad_top, pad_bot), (pad_left, pad_right)))

    X_h, X_w = X.shape
    z_h = X_h - K_h + 1
    z_w = X_w - K_w + 1

    i0 = np.repeat(np.arange(K_h), K_w).reshape((-1, 1))
    i1 = np.repeat(np.arange(z_h), z_w).reshape((1, -1))
    j0 = np.tile(np.arange(K_w), K_h).reshape((-1, 1))
    j1 = np.tile(np.arange(z_w), z_h).reshape((1, -1))

    i = i0 + i1
    j = j0 + j1

    Xp = X[i,j]
    K = np.ravel(K)

    conv_flat = K @ Xp
    conv = conv_flat.reshape((z_h, z_w))

    return conv


X_val = correlate_valid(X, K)
X_same = correlate_same(X, K)

plt.figure()
plt.imshow(X_val)
plt.savefig("our_val.pdf")

plt.figure()
plt.imshow(X_same)
plt.savefig("our_same.pdf")