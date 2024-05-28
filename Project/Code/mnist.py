from tensorflow.keras import datasets
import numpy as np
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from network import Network
import time
from funcs import *
from scheduler import *
from plotting import * #Various plotting functions, we will use heatmap
from funcs import padding
from copy import copy

data_frac = 0.01


digits = datasets.mnist.load_data(path="mnist.npz")
(x_train, y_train), (x_test, y_test) = digits #The data contains a test and a train set
x_train, x_test = x_train/255.0, x_test/255.0 #Normalising the pixel values to be in [0,1]
x_train = x_train[:][:][0:int(data_frac*len(x_train[:][:]))] #The data contains 60000 samples, 6000 should be enough for our purpose
x_test = x_test[:][:][0:int(data_frac*len(x_test[:][:]))]
y_train = y_train[0:int(data_frac*len(y_train))]
y_test = y_test[0:int(data_frac*len(y_test))]


#Greyscale images should have depth 1
x_train = x_train[:,:,:,np.newaxis]
x_test = x_test[:,:,:,np.newaxis]


#Transforming the labels from a single digit to an array of length 10 with the digit corresponding to the index
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#Create CNN with convolution, maxpool, and fully connected layer
def create_convolutional_neural_network_our_code(cost_func, input_shape, n_hidden_neurons, act_func, scheduler, n_filters):
    model = Network(cost_func, input_shape)
    model.add_Convolution_layer((n_filters, 3, 3, 1), act_func, copy(scheduler))
    model.add_MaxPool_layer(2, 2)
    model.add_Flattened_layer()
    model.add_FullyConnected_layer(n_hidden_neurons, act_func, copy(scheduler))
    model.add_FullyConnected_layer(10, softmax, scheduler)
    return model

#Create CNN with convolution and maxpool layer
def create_convolutional_neural_network_our_code_w_o_n(cost_func, input_shape, n_hidden_neurons, act_func, scheduler, n_filters):
    model = Network(cost_func, input_shape)
    model.add_Convolution_layer((n_filters, 3, 3, 1), act_func, copy(scheduler))
    model.add_AveragePool_layer(2, 2)
    model.add_Flattened_layer()
    model.add_FullyConnected_layer(10, softmax, scheduler)
    return model



data_frac = 0.1 #Fraction of data to be included

digits = datasets.mnist.load_data(path="mnist.npz")
(x_train, y_train), (x_test, y_test) = digits #The data contains a test and a train set
x_train, x_test = x_train/255.0, x_test/255.0 #Normalising the pixel values to be in [0,1]
x_train = x_train[:][:][0:int(data_frac*len(x_train[:][:]))] #The data contains 60000 samples, 6000 should be enough for our purpose
x_test = x_test[:][:][0:int(data_frac*len(x_test[:][:]))]
y_train = y_train[0:int(data_frac*len(y_train))]
y_test = y_test[0:int(data_frac*len(y_test))]

#Greyscale images should have depth 1
x_train = x_train[:,:,:,np.newaxis]
x_test = x_test[:,:,:,np.newaxis]


#Transforming the labels from a single digit to an array of length 10 with the digit corresponding to the index
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

epochs = 100
batches = 100
input_shape = x_train.shape[1:4]
n_filters = 20
n_hidden_neurons = None
n_categories = 10

eta_vals = np.logspace(-4, 0, 5)
lmbd_vals = np.logspace(-4, 0, 5)


activation = "lrelu"
act_func = LRELU

eta0 = -4; eta1 = 0; n_eta = eta1-eta0+1
lam0 = -4; lam1 = 0; n_lam = lam1-lam0+1
etas = np.logspace(eta1, eta0, n_eta)
lams = np.logspace(lam0, lam1, n_lam)

val_accs = np.zeros((n_eta, n_lam))
train_accs = np.zeros((n_eta, n_lam))


for i in range(len(etas)):
    for j in range(len(lams)):
        eta = etas[i]
        lam = lams[j]
        scheduler = Adam(eta, 0.9, 0.999)
        network = create_convolutional_neural_network_our_code_w_o_n(CostLogReg, input_shape, n_hidden_neurons, act_func, scheduler, n_filters)
        scores = network.train(x_train, y_train, x_test, y_test, epochs ,batches , lam)
        train_accs_our = scores["train_accuracy"]
        val_accs_our = scores["val_accuracy"]
        train_accs[i,j] = train_accs_our[-1]
        val_accs[i,j] = val_accs_our[-1]

heatmap(train_accs, xticks=lams, yticks=etas, title=f"Training Accuracy, LRELU", xlabel="$\lambda$", ylabel="$\eta$", filename=f"../Figures/LRELU_TrainAccsAveragePool.pdf")
heatmap(val_accs, xticks=lams, yticks=etas, title=f"Validation Accuracy, LRELU", xlabel="$\lambda$", ylabel="$\eta$", filename=f"../Figures/LRELU_ValAccsAveragePool.pdf")
