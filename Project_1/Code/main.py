import jax.numpy as jnp
from convolution import Convolution
from maxpool import MaxPool
from fullyconnected import FullyConnected
from flatteningfunc import flatten

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

input_fully = flatten(pool_output)

print(input_fully)

input_shape = jnp.shape(input_fully)
output_shape = (input_shape[0], 6)

fc = FullyConnected(input_shape,output_shape)
output = fc.feed_forward(input_fully)

print(output)

target = jnp.array([[0,0,0,0,0,1],[0,0,0,0,0,1]])

print(fc.weights)
fc.backpropagate(output, target, 0.1)
print(fc.weights)
