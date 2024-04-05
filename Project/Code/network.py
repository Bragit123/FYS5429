import numpy as np
from jax import vmap, grad
from funcs import derivate
from sklearn.utils import resample
from convolution import Convolution
from flattenedlayer import FlattenedLayer
from maxpool import MaxPool
from fullyconnected import FullyConnected

class Network:
    def __init__(self, cost_func):
        self.cost_func = cost_func
        self.layers = []
        self.num_layers = 0

    def add_layer(self, layer: Convolution | FlattenedLayer | MaxPool | FullyConnected):
        self.layers.append(layer)
        self.num_layers += 1

    def reset_weights(self, seed):
        for layer in self.layers:
            layer.reset_weights(seed)
    
    def reset_schedulers(self):
        for layer in self.layers:
            layer.reset_schedulers()

    def feed_forward(self, input: np.ndarray):
        layer_output = self.layers[0].feed_forward(input)
        for i in range(1, self.num_layers):
            layer_output = self.layers[i].feed_forward(layer_output)

        return layer_output

    def predict(self, input: np.ndarray):
        output = self.feed_forward(input)
        predicted = np.where(output > 0.5, 1, 0)
        return predicted

    def backpropagate(self, output, target, lmbd = 0.1):
        grad_cost = derivate(self.cost_func(target))
        dC_doutput = grad_cost(output)

        #print(-(1.0 / target.shape[0]) * (target/(output+10**(-10))-(1-target)/(1-output+10**(-10))))
        #print(dC_doutput)


        for i in range(self.num_layers-1, -1, -1):
            dC_doutput = self.layers[i].backpropagate(dC_doutput, lmbd = lmbd)

    def train(self, input_train, target_train, input_val = None, target_val = None, epochs=100, batches=1, lmbd = 0.1, seed=100):
        self.reset_weights(seed) # Reset weights for new training
        batch_size = input_train.shape[0] // batches

        train_cost = self.cost_func(target_train)
        train_error = np.zeros(epochs)
        train_accuracy = np.zeros(epochs)

        if input_val is not None:
            val_cost = self.cost_func(target_val)
            val_error = np.zeros(epochs)
            val_accuracy = np.zeros(epochs)

        input_train, target_train = resample(input_train, target_train, replace=False)

        for e in range(epochs):
            print("EPOCH: " + str(e+1) + "/" + str(epochs), end="\r")
            for b in range(batches):
                if b == batches - 1:
                    input_batch = input_train[b * batch_size :]
                    target_batch = target_train[b * batch_size :]
                else:
                    input_batch = input_train[b * batch_size : (b+1) * batch_size]
                    target_batch = target_train[b * batch_size : (b+1) * batch_size]

                output_batch = self.feed_forward(input_batch)
                self.backpropagate(output_batch, target_batch, lmbd)

            self.reset_schedulers()
            
            train_output = self.feed_forward(input_train)
            train_predict = self.predict(input_train)
            train_error[e] = train_cost(train_predict)
            train_accuracy[e] = np.mean(train_predict == target_train)

            if input_val is not None:
                val_predict = self.predict(input_val)
                val_error[e] = val_cost(val_predict)
                val_accuracy[e] = np.mean(val_predict == target_val)
                val_output = self.feed_forward(input_val)


        scores = {
            "train_error": train_error,
            "train_accuracy": train_accuracy,
            "train_predict": train_predict,
            "train_output": train_output
        }
        if input_val is not None:
            scores["val_error"] = val_error
            scores["val_accuracy"] = val_accuracy
            scores["val_predict"] = val_predict
            scores["val_output"] = val_output

        return scores
