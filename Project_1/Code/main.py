import jax.numpy as jnp
from jax import vmap
from convolution import Convolution
from maxpool import MaxPool
from fullyconnected import FullyConnected
from flatteningfunc import Flattened_Layer
from funcs import CostCrossEntropy, derivate

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

image_shape = jnp.shape(image)[1:]
kernel_shape = (2,3,2,2)

cnn = Convolution(image_shape, kernel_shape)
output = cnn.feed_forward(image)

# print(output)

pool = MaxPool(output.shape, jnp.array([2,2]), 1)
pool_output = pool.feed_forward(output)

# print(pool_output)

flat = Flattened_Layer()
input_fully = flat.feed_forward(pool_output)

# print(input_fully)

input_length = jnp.shape(input_fully)[1]
output_length = 6

fc = FullyConnected(input_length,output_length)
output = fc.feed_forward(input_fully)

# print(output)

target = jnp.array([[0,0,0,0,0,1],[0,0,0,0,0,1]])

# print(fc.weights)
gradCost = vmap(vmap(derivate(CostCrossEntropy(target))))
dCdoutput = gradCost(output)
grad_input = fc.backpropagate(input_fully, dCdoutput, 0.1)
# print(fc.weights)
# print(grad_input)
