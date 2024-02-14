import jax.numpy as jnp
from convolution import Convolution
from maxpool import MaxPool

image = jnp.array([
    [
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
    ], [
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
    ]
])

image_shape = jnp.shape(image)
kernel_shape = (2,3,2,2)

cnn = Convolution(image_shape, kernel_shape)
output = cnn.feed_forward(image)

print(output)

pool = MaxPool(output.shape, jnp.array([2,2]), 1)
pool_output = pool.feed_forward(output)

print(pool_output)