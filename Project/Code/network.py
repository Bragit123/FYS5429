import numpy as np
from funcs import derivate
from sklearn.utils import resample
from layer import Layer
from convolution import Convolution
from maxpool import MaxPool
from averagepool import AveragePool
from flattenedlayer import FlattenedLayer
from fullyconnected import FullyConnected

from funcs import RELU, sigmoid, derivate
from copy import copy 

class Network:
    def __init__(self, cost_func, input_shape: tuple | int, seed: int = 100):
        self.seed = seed
        self.cost_func = cost_func
        self.layers: list[Layer] = []
        self.num_layers = 0
        self.input_shape = input_shape

    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        self.num_layers += 1

    def reset_weights(self):
        for layer in self.layers:
            layer.reset_weights()
    
    def reset_schedulers(self):
        for layer in self.layers:
            layer.reset_schedulers()

    def feed_forward(self, input: np.ndarray) -> np.ndarray:
        layer_output = self.layers[0].feed_forward(input)
        for i in range(1, self.num_layers):
            layer_output = self.layers[i].feed_forward(layer_output)

        return layer_output

    def predict(self, input: np.ndarray):
        output = self.feed_forward(input)
        predicted = np.zeros_like(output)
        ind_1_axis = np.argmax(output,axis=1)

        #or i in range(predicted.shape[0]):
        ind_0_axis = np.arange(predicted.shape[0])
        predicted[ind_0_axis, ind_1_axis] = 1
        return predicted

    def backpropagate(self, output, target, lmbd = 0.1):
        grad_cost = derivate(self.cost_func(target))
        dC_doutput = grad_cost(output)

        #print(-(1.0 / target.shape[0]) * (target/(output+10**(-10))-(1-target)/(1-output+10**(-10))))
        #print(dC_doutput)


        for i in range(self.num_layers-1, -1, -1):
            dC_doutput = self.layers[i].backpropagate(dC_doutput, lmbd = lmbd)

    def train(self, input_train, target_train, input_val = None, target_val = None, epochs=100, batches=1, lmbd = 0.1, seed=100):
        self.reset_weights() # Reset weights for new training
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
            print("EPOCH: " + str(e+1) + "/" + str(epochs))
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
            train_accuracy[e] = np.mean(np.all(train_predict == target_train, axis = 1))

            #print(np.all(train_predict == target_train, axis = 1))

            if input_val is not None:
                val_predict = self.predict(input_val)
                val_error[e] = val_cost(val_predict)
                val_accuracy[e] = np.mean(np.all(val_predict == target_val, axis = 1))
                #print(np.all(val_predict == target_val, axis = 1))
                #print(val_accuracy[e])
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
    
    def next_layer_input_shape(self):
        if self.num_layers == 0:
            input_shape = self.input_shape
        
        else:
            prev_layer = self.layers[-1]
            input_shape = prev_layer.find_output_shape()
        
        return input_shape

    ## Methods for adding layers
    def add_Convolution_layer(self, kernel_size: tuple, act_func, scheduler):
        input_shape = self.next_layer_input_shape()
        cnn_layer = Convolution(input_shape, kernel_size, act_func, copy(scheduler), self.seed)

        self.layers.append(cnn_layer)
        self.num_layers += 1
    
    def add_MaxPool_layer(self, scale_factor, stride):
        input_shape = self.next_layer_input_shape()
        maxpool_layer = MaxPool(input_shape, scale_factor, stride, self.seed)

        self.layers.append(maxpool_layer)
        self.num_layers += 1
    
    def add_AveragePool_layer(self, scale_factor, stride):
        input_shape = self.next_layer_input_shape()
        averagepool_layer = AveragePool(input_shape, scale_factor, stride, self.seed)

        self.layers.append(averagepool_layer)
        self.num_layers += 1
    
    def add_Flattened_layer(self):
        input_shape = self.next_layer_input_shape()
        flattened_layer = FlattenedLayer(input_shape, self.seed)

        self.layers.append(flattened_layer)
        self.num_layers += 1

    def add_FullyConnected_layer(self, output_length, act_func, scheduler):
        input_shape = self.next_layer_input_shape()
        fc_layer = FullyConnected(input_shape, output_length, act_func, scheduler, self.seed)
        
        self.layers.append(fc_layer)
        self.num_layers += 1
                
