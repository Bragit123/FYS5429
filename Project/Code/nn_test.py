import numpy as np
import matplotlib.pyplot as plt
from funcs import CostLogReg, sigmoid, CostCrossEntropy, softmax
from network import Network
from fullyconnected import FullyConnected
from scheduler import *
from copy import copy

from sklearn import datasets
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

# Load cancer dataset
cancer = datasets.load_breast_cancer()
X = cancer.data
t = cancer.target
t = np.c_[t]

X = minmax_scale(X, feature_range=(0,1), axis=0)

X_train, X_val, t_train, t_val = train_test_split(X, t, test_size=0.2, random_state=100)

# Set values for neural network
act_func = sigmoid
cost_func = CostLogReg

# Set constants
rho = 0.9
rho2 = 0.999
momentum = 0.01
batches = 10
epochs = 100

# Layers
X_shape = np.shape(X)
t_shape = np.shape(t)
n_nodes_hidden = 100

etas = [0.01]
lmds = [0.001]
for eta in etas:
    for lmbd in lmds:
        #scheduler = AdamMomentum(eta, 0.9, 0.999, 0.01) #temporary
        scheduler = Adam(eta, rho, rho2) #temporary
        #scheduler = AdagradMomentum(0.01, 0.001) #temporary
        #scheduler = Constant(0.1) #temporary
        #scheduler = Momentum(eta, 0.01)

        #scheduler = RMS_prop(0.01, 0.99)

        input_layer = FullyConnected(X_shape[1], n_nodes_hidden, act_func, copy(scheduler))
        # hidden_layer = FullyConnected(n_nodes_hidden, n_nodes_hidden, sigmoid)
        output_layer = FullyConnected(n_nodes_hidden, t_shape[1], act_func, copy(scheduler))

        # Create network
        network = Network(cost_func)
        network.add_layer(input_layer)
        # network.add_layer(hidden_layer)
        network.add_layer(output_layer)

        # pred = network.feed_forward(X)
        # print(pred)
        # Train network
        scores = network.train(X_train, t_train, X_val, t_val, epochs=epochs, batches=batches, lmbd = lmbd)
        epoch_arr = np.arange(epochs)

        plt.figure()
        plt.title("Accuracies")
        plt.plot(epoch_arr, scores["train_accuracy"], label="Training data")
        plt.plot(epoch_arr, scores["val_accuracy"], label="Validation data")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("accuracy.pdf")
