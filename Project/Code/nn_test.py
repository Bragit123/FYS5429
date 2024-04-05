import numpy as np
import matplotlib.pyplot as plt
from funcs import CostLogReg, sigmoid, CostCrossEntropy, softmax
from network import Network
from fullyconnected import FullyConnected
from scheduler import *
from copy import copy
import plotting

from sklearn import datasets
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import numpy as np

# Load cancer dataset
cancer = datasets.load_breast_cancer()
X = cancer.data
<<<<<<< HEAD:Project_1/Code/test.py
X = X
X = minmax_scale(X, feature_range=(0,1), axis=0)
=======
>>>>>>> 5836ba58fe18a7d53b99024983bafb71ad40702f:Project/Code/nn_test.py
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

<<<<<<< HEAD:Project_1/Code/test.py


eta0 = -4; eta1 = -1; n_eta = eta1-eta0+1
lam0 = -5; lam1 = -1; n_lam = lam1-lam0+1
etas = np.logspace(eta0, eta1, n_eta)
lmds = np.logspace(lam0, lam1, n_lam)
train_accs = np.zeros((4,5))
val_accs = np.zeros((4,5))

#etas = [0.01]
#lmds = [10**(-5)]
rho = 0.9
rho2 = 0.999
momentum = 0.01
batches = 10
epochs = 50

for i in range(len(etas)):
    for j in range(len(lmds)):
        #scheduler = AdamMomentum(eta, 0.9, 0.999, 0.01) #temporary
        scheduler = Adam(etas[i], rho, rho2) #temporary
=======
etas = [0.01]
lmds = [0.001]
for eta in etas:
    for lmbd in lmds:
        #scheduler = AdamMomentum(eta, 0.9, 0.999, 0.01) #temporary
        scheduler = Adam(eta, rho, rho2) #temporary
>>>>>>> 5836ba58fe18a7d53b99024983bafb71ad40702f:Project/Code/nn_test.py
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
<<<<<<< HEAD:Project_1/Code/test.py
        print(f"eta: {etas[i]}, lmbd: {lmds[j]}")
        scores = network.train(X_train, t_train, X_val, t_val, epochs=epochs, batches=batches, lmbd = lmds[j])
        print(np.argmax(scores["train_accuracy"]))
        epoch_arr = jnp.arange(epochs)
        train_accs[i,j] = scores["train_accuracy"][-1]
        val_accs[i,j] = scores["val_accuracy"][-1]
=======
        scores = network.train(X_train, t_train, X_val, t_val, epochs=epochs, batches=batches, lmbd = lmbd)
        epoch_arr = np.arange(epochs)
>>>>>>> 5836ba58fe18a7d53b99024983bafb71ad40702f:Project/Code/nn_test.py

        if i== 2 and j == 0:  #Best accuracy
            plt.figure()
            plt.title("Accuracies")
            plt.plot(epoch_arr, scores["train_accuracy"], label="Training data")
            plt.plot(epoch_arr, scores["val_accuracy"], label="Validation data")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig("accuracy.pdf")

title = "Accuracies train"
filename = "heatmap_train.pdf"
plotting.heatmap(data=train_accs, xticks=lmds, yticks=etas, title=title, xlabel="$\\lambda$", ylabel="$\\eta$", filename=filename)
title = "Accuracies validation"
filename = "heatmap_val.pdf"
plotting.heatmap(data=val_accs, xticks=lmds, yticks=etas, title=title, xlabel="$\\lambda$", ylabel="$\\eta$", filename=filename)
