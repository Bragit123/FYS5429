import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp

class Flattened_Layer:
    def __init__(self):
        return None
    
    def feed_forward(self, input):
        input_shape = input.shape
        length_flattened = input_shape[1]*input_shape[2]*input_shape[3]

        flattened_output = jnp.reshape(input, (input_shape[0], length_flattened))

        return flattened_output
