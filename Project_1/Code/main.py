import jax.numpy as jnp
from convolution import Convolution

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
output_shape = (kernel_shape[0], image_shape[1]-kernel_shape[2]+1, image_shape[2]-kernel_shape[3]+1)
output = jnp.zeros(output_shape)

cnn = Convolution(image_shape, kernel_shape)
output = cnn.feed_forward(image)

print(output)