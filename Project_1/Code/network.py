import jax.numpy as jnp
from jax import vmap, grad
from funcs import derivate
from convolution import Convolution
from flatteningfunc import Flattened_Layer
from maxpool import MaxPool
from fullyconnected import FullyConnected

class Network:
    def __init__(self, cost_func):
        self.cost_func = cost_func
        self.layers = []
        self.num_layers = 0

    def add_layer(self, layer: Convolution | Flattened_Layer | MaxPool | FullyConnected):
        self.layers.append(layer)
        self.num_layers += 1

    def reset_weights(self, seed):
        for layer in self.layers:
            layer.reset_weights(seed)

    def feed_forward(self, input: jnp.ndarray):
        layer_output = self.layers[0].feed_forward(input)
        for i in range(1, self.num_layers):
            layer_output = self.layers[i].feed_forward(layer_output)

        return layer_output

    def predict(self, input: jnp.ndarray):
        output = self.feed_forward(input)
        predicted = jnp.where(output > 0.5, 1, 0)
        return predicted

    def backpropagate(self, output, target):
        grad_cost = vmap(vmap(derivate(self.cost_func(target))))
        dC_doutput = grad_cost(output)

        #print(-(1.0 / target.shape[0]) * (target/(output+10**(-10))-(1-target)/(1-output+10**(-10))))
        #print(dC_doutput)


        for i in range(self.num_layers-1, -1, -1):
            dC_doutput = self.layers[i].backpropagate(dC_doutput)

    def train(self, input_train, target_train, input_val = None, target_val = None, epochs=100, batches=1, seed=100):
        self.reset_weights(seed) # Reset weights for new training
        batch_size = input_train.shape[0] // batches

        train_cost = self.cost_func(target_train)
        train_error = jnp.zeros(epochs)
        train_accuracy = jnp.zeros(epochs)

        if input_val is not None:
            val_cost = self.cost_func(target_val)
            val_error = jnp.zeros(epochs)
            val_accuracy = jnp.zeros(epochs)

        for e in range(epochs):
            print("EPOCH: " + str(e+1) + "/" + str(epochs))
            for b in range(batches):
                if b == batches - 1:
                    input_batch = input_train[b * batch_size :]
                    target_batch = target_train[b * batch_size :]
                else:
                    input_batch = input_train[b * batch_size : (b+1) * batch_size]
                    target_batch = target_train[b * batch_size : (b+1) * batch_size]

                output_batch = self.feed_forward(input_batch)
                self.backpropagate(output_batch, target_batch)

            train_predict = self.predict(input_train)
            train_error = train_error.at[e].set(train_cost(train_predict))
            train_accuracy = train_accuracy.at[e].set(jnp.mean(train_predict == target_train))

            if input_val is not None:
                val_predict = self.predict(input_val)
                val_error = val_error.at[e].set(val_cost(val_predict))
                val_accuracy = val_accuracy.at[e].set(jnp.mean(val_predict == target_val))

        scores = {
            "train_error": train_error,
            "train_accuracy": train_accuracy,
        }
        if input_val is not None:
            scores["val_error"] = val_error
            scores["val_accuracy"] = val_accuracy

        return scores
