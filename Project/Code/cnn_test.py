from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
import numpy as np
import matplotlib.pyplot as plt
from network import Network
from convolution import Convolution
from fullyconnected import FullyConnected
from flattenedlayer import FlattenedLayer
from maxpool import MaxPool
from scheduler import Adam
from funcs import CostLogReg, sigmoid, RELU
from copy import copy
# from plotting import * #Various plotting functions, we will use heatmap

digits = datasets.mnist.load_data(path="mnist.npz")
(x_train, y_train), (x_test, y_test) = digits #The data contains a test and a train set

x_train, x_test = x_train/255.0, x_test/255.0 #Normalising the pixel values to be in [0,1]
x_train = x_train[:][:][0:int(0.01*len(x_train[:][:]))] #The data contains 60000 samples, 6000 should be enough for our purpose
x_test = x_test[:][:][0:int(0.01*len(x_test[:][:]))]
y_train = y_train[0:int(0.01*len(y_train))]
y_test = y_test[0:int(0.01*len(y_test))]
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#Greyscale images should have depth 1
x_train = x_train[:,np.newaxis,:,:]
x_test = x_test[:,np.newaxis,:,:]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#Transforming the labels from a single digit to an array of length 10 with the
#digit corresponding to the index

# Parameters
input_size = x_train.shape[1:]
kernel_size = (1, 1, 3, 3)
pool_input_size = (1, 28-3+1, 28-3+1)
input_size2 = (1, 13, 13)
pool_input_size2 = (1, 9, 9)
scale_factor = 2; stride = 2
fc_size = 4*4

cost_func = CostLogReg
act_func = sigmoid
scheduler = Adam(0.1, 0.9, 0.999)

# Layers
# conv = Convolution(input_size, kernel_size)
# pool = MaxPool(pool_input_size, scale_factor, stride)
# flat = FlattenedLayer()
# fc = FullyConnected(fc_size, 20, act_func, scheduler)
# out = FullyConnected(20, 10, act_func, scheduler)

# Network
network = Network(cost_func, input_size)
# network.add_layer(conv)
# network.add_layer(pool)
# network.add_layer(flat)
# network.add_layer(fc)
# network.add_layer(out)

network.add_Convolution_layer(kernel_size)
network.add_MaxPool_layer(scale_factor, stride)
#network.add_Convolution_layer(kernel_size)
#network.add_MaxPool_layer(scale_factor, stride)
network.add_Flattened_layer()
network.add_FullyConnected_layer(20, act_func, copy(scheduler))
network.add_FullyConnected_layer(10, act_func, copy(scheduler))

epochs = 10
batches = 1
lmbd = 0.001
scores = network.train(x_train, y_train, x_test, y_test, epochs, batches, lmbd)

epoch_arr = np.arange(epochs)
plt.figure()
plt.title("Accuracies")
plt.plot(epoch_arr, scores["train_accuracy"], label="Training data")
plt.plot(epoch_arr, scores["val_accuracy"], label="Validation data")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("cnn_accuracy.pdf")
