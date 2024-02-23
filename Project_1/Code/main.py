import jax.numpy as jnp
from jax import vmap
from network import Network
from convolution import Convolution
from maxpool import MaxPool
from fullyconnected import FullyConnected
from flatteningfunc import Flattened_Layer
from funcs import CostLogReg, CostCrossEntropy, derivate, CostOLS

input = jnp.array([
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

input = jnp.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
target = jnp.array([
    [0],
    [1],
    [1],
    [0]
])

cost_func = CostLogReg

cnn = Network(cost_func)
cnn.add_layer(FullyConnected(input_length=2, output_length=2))
cnn.add_layer(FullyConnected(input_length=2, output_length=1))

print(cnn.feed_forward(input))
cnn.train(input, target, epochs=500, batches=1, seed=100)
print(cnn.feed_forward(input))


# image_shape = jnp.shape(image)[1:]
# kernel_shape = (2,3,2,2)

# cnn = Convolution(image_shape, kernel_shape)
# output = cnn.feed_forward(image)

# # print(output)

# pool = MaxPool(output.shape, jnp.array([2,2]), 1)
# pool_output = pool.feed_forward(output)

# # print(pool_output)

# flat = Flattened_Layer()
# input_fully = flat.feed_forward(pool_output)

# # print(input_fully)

# input_length = jnp.shape(input_fully)[1]
# output_length = 6

# fc = FullyConnected(input_length,output_length)
# output_fully = fc.feed_forward(input_fully)

# # print(output)

# target = jnp.array([[0,0,0,0,0,1],[0,0,0,0,0,1]])

# # print(fc.weights)
# gradCost = vmap(vmap(derivate(CostCrossEntropy(target))))
# dCdoutput = gradCost(output_fully)
# lmbd = 0.1

# grad_fc = fc.backpropagate(dCdoutput, lmbd)

# grad_flat = flat.backpropagate(grad_fc, lmbd)

# grad_pool = pool.backpropagate(grad_flat, lmbd)

# grad_cnn = cnn.backpropagate(grad_pool, lmbd)

# print(jnp.shape(grad_fc), jnp.shape(input_fully))
# print(jnp.shape(grad_flat), jnp.shape(pool_output))
# print(jnp.shape(grad_pool), jnp.shape(output))
# print(jnp.shape(grad_cnn), jnp.shape(image))
