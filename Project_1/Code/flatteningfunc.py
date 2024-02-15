import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp

def flatten(input):
    input_shape = input.shape
    length_flattened = input_shape[1]*input_shape[2]*input_shape[3]
    flattened_output = jnp.zeros((input_shape[0], length_flattened))

    for i in range(input_shape[0]):
        flattened_output = flattened_output.at[i,:].set(input[i].flatten())

    return flattened_output
