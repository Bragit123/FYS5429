from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
import numpy as np
import matplotlib.pyplot as plt
from network import Network
from convolution import Convolution
from fullyconnected import FullyConnected
from flattenedlayer import FlattenedLayer
from maxpool import MaxPool
from scheduler import Adam, Momentum, Constant
from funcs import CostLogReg, sigmoid, LRELU, softmax, padding, CostCrossEntropy
from copy import copy
from plotting import * #Various plotting functions, we will use heatmap

digits = datasets.mnist.load_data(path="mnist.npz")
(x_train, y_train), (x_test, y_test) = digits #The data contains a test and a train set

x_train, x_test = x_train/255.0, x_test/255.0 #Normalising the pixel values to be in [0,1]
x_train = x_train[:][:][0:int(0.01*len(x_train[:][:]))] #The data contains 60000 samples, 6000 should be enough for our purpose
x_test = x_test[:][:][0:int(0.01*len(x_test[:][:]))]
y_train = y_train[0:int(0.01*len(y_train))]
y_test = y_test[0:int(0.01*len(y_test))]
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#Greyscale images should have depth 1
x_train = x_train[:,:,:,np.newaxis]
x_test = x_test[:,:,:,np.newaxis]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#Transforming the labels from a single digit to an array of length 10 with the
#digit corresponding to the index

# Parameters
input_size = x_train.shape[1:]

#x_train = padding(x_train)
#x_test = padding(x_test)

kernel_size = (2, 3, 3, 1)
scale_factor = 2; stride = 1


cost_func = CostLogReg
act_func = sigmoid
output_act = softmax

epochs = 50
batches = 15
eta0 = -3; eta1 = -1; n_eta = eta1-eta0+1
lam0 = -4; lam1 = -2; n_lam = lam1-lam0+1
etas = np.logspace(eta0, eta1, n_eta)
lmds = np.logspace(lam0, lam1, n_lam)

train_accs = np.zeros((n_eta, n_lam))
val_accs = np.zeros((n_eta, n_lam))

rho = 0.9
rho2 = 0.999
"""
for i in range(len(etas)):
    for j in range(len(lmds)):
        #scores = network.train(x_train, y_train, x_test, y_test, epochs, batches, lmbd)

        scheduler = Adam(eta=etas[i], rho = 0.9, rho2 = 0.999)  #temporary
        #scheduler = AdagradMomentum(0.01, 0.001) #temporary
        #scheduler = Constant(0.1) #temporary
        #scheduler = Momentum(eta, 0.01)

        #scheduler = RMS_prop(0.01, 0.99)

        #input_layer = FullyConnected(X_shape[1], n_nodes_hidden, act_func, copy(scheduler))
        # hidden_layer = FullyConnected(n_nodes_hidden, n_nodes_hidden, sigmoid)
        #output_layer = FullyConnected(n_nodes_hidden, t_shape[1], act_func, copy(scheduler))

        # Create network
        network = Network(cost_func, input_size)
        # network.add_layer(conv)
        # network.add_layer(pool)
        # network.add_layer(flat)
        # network.add_layer(fc)
        # network.add_layer(out)

        network.add_Convolution_layer(kernel_size, act_func, copy(scheduler), stride=2)
        network.add_MaxPool_layer(scale_factor, stride)
        #network.add_Convolution_layer(kernel_size)
        #network.add_MaxPool_layer(scale_factor, stride)
        network.add_Flattened_layer()
        network.add_FullyConnected_layer(50, act_func, copy(scheduler))
        network.add_FullyConnected_layer(10, act_func, copy(scheduler))


        # pred = network.feed_forward(X)
        # print(pred)
        # Train network
        print(f"eta: {etas[i]}, lmbd: {lmds[j]}")
        scores = network.train(x_train, y_train, x_test, y_test, epochs=epochs, batches=batches, lmbd = lmds[j])
        print(np.argmax(scores["train_accuracy"]))
        epoch_arr = np.arange(epochs)
        train_accs[i,j] = scores["train_accuracy"][-1]
        val_accs[i,j] = scores["val_accuracy"][-1]
        plt.figure()
        plt.title("Accuracies")
        plt.plot(epoch_arr, scores["train_accuracy"], label="Training data")
        plt.plot(epoch_arr, scores["val_accuracy"], label="Validation data")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("cnn_accuracy.pdf")

        epoch_arr = np.arange(epochs)
        if (i==0 and j==0):
            plt.figure()
            plt.title("Accuracies")
            plt.plot(epoch_arr, scores["train_accuracy"], label="Training data")
            plt.plot(epoch_arr, scores["val_accuracy"], label="Validation data")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig("cnn_accuracy.pdf")

    
title = "Accuracies train"
filename = "heatmap_train_cnn.pdf"
heatmap(data=train_accs, xticks=lmds, yticks=etas, title=title, xlabel="$\\lambda$", ylabel="$\\eta$", filename=filename)
title = "Accuracies validation"
filename = "heatmap_val_cnn.pdf"
heatmap(data=val_accs, xticks=lmds, yticks=etas, title=title, xlabel="$\\lambda$", ylabel="$\\eta$", filename=filename)
"""
eta = 0.001
lmbd = 0.0001
scheduler = Adam(eta=eta, rho = 0.9, rho2 = 0.999) 
#scheduler = AdagradMomentum(0.01, 0.001) #temporary
#scheduler = Constant(0.1) #temporary
#scheduler = Momentum(eta, 0.01)

#scheduler = RMS_prop(0.01, 0.99)

#input_layer = FullyConnected(X_shape[1], n_nodes_hidden, act_func, copy(scheduler))
# hidden_layer = FullyConnected(n_nodes_hidden, n_nodes_hidden, sigmoid)
#output_layer = FullyConnected(n_nodes_hidden, t_shape[1], act_func, copy(scheduler))

# Create network
network = Network(cost_func, input_size)
# network.add_layer(conv)
# network.add_layer(pool)
# network.add_layer(flat)
# network.add_layer(fc)
# network.add_layer(out)

network.add_Convolution_layer(kernel_size, act_func, copy(scheduler), stride=2)
network.add_MaxPool_layer(scale_factor, stride)
#network.add_Convolution_layer(kernel_size)
#network.add_MaxPool_layer(scale_factor, stride)
network.add_Flattened_layer()
network.add_FullyConnected_layer(50, act_func, copy(scheduler))
network.add_FullyConnected_layer(10, act_func, copy(scheduler))


# pred = network.feed_forward(X)
# print(pred)
# Train network
scores = network.train(x_train, y_train, x_test, y_test, epochs=epochs, batches=batches, lmbd = lmbd)
print(np.argmax(scores["train_accuracy"]))
epoch_arr = np.arange(epochs)
plt.figure()
plt.title("Accuracies")
plt.plot(epoch_arr, scores["train_accuracy"], label="Training data")
plt.plot(epoch_arr, scores["val_accuracy"], label="Validation data")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("cnn_accuracy.pdf")
"""
