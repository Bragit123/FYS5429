import jax.numpy as jnp
import matplotlib.pyplot as plt
from funcs import CostLogReg, sigmoid, CostCrossEntropy
from network import Network
from fullyconnected import FullyConnected
from scheduler import *
from copy import copy

from sklearn import datasets
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

# Cost function
cost_func = CostLogReg
act_func = sigmoid

# Load dataset
cancer = datasets.load_breast_cancer()
X = cancer.data
X = minmax_scale(X, feature_range=(0,1), axis=0)
t = cancer.target
t = jnp.c_[t]
X_train, X_val, t_train, t_val = train_test_split(X, t, test_size=0.2, random_state=100)

# Layers
X_shape = jnp.shape(X)
t_shape = jnp.shape(t)
n_nodes_hidden = 100

scheduler = AdamMomentum(0.01, 0.9, 0.999, 0.001) #temporary
# scheduler = Adam(0.01, 0.9, 0.999) #temporary
# scheduler = AdagradMomentum(0.01, 0.001) #temporary
# scheduler = Constant(0.1) #temporary

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
epochs = 200
scores = network.train(X_train, t_train, X_val, t_val, epochs=epochs)
print(jnp.argmax(scores["train_accuracy"]))
epoch_arr = jnp.arange(epochs)

plt.figure()
plt.title("Accuracies")
plt.plot(epoch_arr, scores["train_accuracy"], label="Training data")
plt.plot(epoch_arr, scores["val_accuracy"], label="Validation data")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy.pdf")
