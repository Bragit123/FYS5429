import tensorflow as tf   
 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)


from network import Network
import time
from funcs import *
from scheduler import *
from plotting import * #Various plotting functions, we will use heatmap
from funcs import padding
from copy import copy

import os
os.environ['SSL_CERT_FILE'] = ''

data_frac = 0.01

 
# Distribute it to train and test set
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary

filename = "cifar10data/data_batch_"

x_train = np.zeros((50000,32,32,3))
x_test = np.zeros((10000,32,32,3))

y_train = np.zeros(50000)
y_test = np.zeros(10000)

for i in range(0,5):
    train_dict = unpickle(filename + f"{i+1}")
    x_train[i*10000:(i+1)*10000,:,:,:] = np.array(train_dict[b"data"].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1))
    y_train[i*10000:(i+1)*10000] = np.array(train_dict[b"labels"])


test_dict = unpickle("cifar10data/test_batch")

x_test = np.array(test_dict[b"data"].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1))
y_test = np.array(test_dict[b"labels"])

#Transforming the labels from a single digit to an array of length 10 with the digit corresponding to the index
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_test.shape)
print(y_test.shape)
print(x_train.shape)
print(y_train.shape)

# Reduce pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0
#Display random image
plt.imshow(x_train[10232])
plt.axis('off')  # Turn off axis labels
plt.show()

 
x_train = x_train[:][:][0:int(data_frac*len(x_train[:][:]))]
x_test = x_test[:][:][0:int(data_frac*len(x_test[:][:]))]
y_train = y_train[0:int(data_frac*len(y_train))]
y_test = y_test[0:int(data_frac*len(y_test))]





def create_convolutional_neural_network_keras(input_shape, receptive_field,
                                              n_filters, n_hidden_neurons, n_categories,
                                              eta, lmbd, activation, rho, rho2):
    model = Sequential()
    model.add(layers.Conv2D(6, (4, 4), padding='valid', activation='leaky_relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    #model.add(layers.Dense(20, activation='leaky_relu'))
    model.add(layers.Dense(10, activation='softmax'))

    adam = optimizers.Adam(eta, rho, rho2)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def create_convolutional_neural_network_our_code(cost_func, input_shape, n_hidden_neurons, act_func, scheduler, n_filters):
    model = Network(cost_func, input_shape)
    model.add_Convolution_layer((6, 4, 4, 3), act_func, copy(scheduler))
    model.add_MaxPool_layer(2, 2)
    #model.add_Convolution_layer((8, 3, 3, 6), act_func, copy(scheduler))
    #model.add_MaxPool_layer(2, 2)

    model.add_Flattened_layer()
    #model.add_FullyConnected_layer(20, act_func, scheduler)
    model.add_FullyConnected_layer(10, softmax, scheduler)
    return model


epochs = 50
batch_size = 50
batches = x_train.shape[0] // batch_size
input_shape = x_train.shape[1:4]
receptive_field = 3
n_filters = 10
n_hidden_neurons= 50
n_categories = 10

eta_vals = np.logspace(-5, -2, 4)
lmbd_vals = np.logspace(-5, -2, 4)

train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
activation = "leaky_relu"
act_func = LRELU

rho = 0.9
rho2 = 0.999


for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):

        CNN = create_convolutional_neural_network_keras(input_shape, receptive_field,
                                                n_filters, n_hidden_neurons, n_categories,
                                                eta, lmbd, activation, rho, rho2)
        history = CNN.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0)

        train_accuracy[i][j] = CNN.evaluate(x_train, y_train)[1]
        test_accuracy[i][j] = CNN.evaluate(x_test, y_test)[1]
        print("Learning rate = ", eta)
        print("Lambda = ", lmbd)
        print(f"Test accuracy: {test_accuracy[i][j]:.3f}")
        print()


        val_accs_tf = history.history["val_accuracy"]
        
        epoch_arr = np.arange(epochs)
        plt.figure()
        plt.title("Validation accuracies")
        plt.plot(epoch_arr, val_accs_tf, label="Tensorflow")
        plt.legend()
        plt.savefig("tf_accs_cifar10.pdf")
        #Plotting the training and test accuracy
# Plotting the training and test accuracy
heatmap(train_accuracy, xticks=lmbd_vals, yticks=eta_vals, title=f"Training Accuracy, Leaky ReLU", xlabel="$\lambda$", ylabel="$\eta$", filename=f"cifar10_train_small_500.pdf")
heatmap(test_accuracy, xticks=lmbd_vals, yticks=eta_vals, title=f"Validation Accuracy, Leaky ReLU", xlabel="$\lambda$", ylabel="$\eta$", filename=f"cifar10_test_small_500.pdf")



for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        scheduler = Adam(eta, 0.9, 0.999)
        CNN = create_convolutional_neural_network_our_code(CostLogReg, input_shape, n_hidden_neurons, act_func, scheduler, n_filters)
        scores = CNN.train(x_train, y_train, x_test, y_test, epochs, batches, lmbd)

        train_accs_our = scores["train_accuracy"]
        val_accs_our = scores["val_accuracy"]

        train_accuracy[i][j] = train_accs_our[-1]
        test_accuracy[i][j] = val_accs_our[-1]
        print("Learning rate = ", eta)
        print("Lambda = ", lmbd)
        print(f"Test accuracy: {test_accuracy[i][j]:.3f}")
        print()
        
        plt.figure()
        epoch_arr = np.arange(epochs)
        plt.figure()
        plt.title("Validation accuracies")
        plt.plot(epoch_arr, val_accs_our, label="Our network")
        plt.legend()
        plt.savefig("our_accs_cifar10.pdf")
        #Plotting the training and test accuracy
# Plotting the training and test accuracy
heatmap(train_accuracy, xticks=lmbd_vals, yticks=eta_vals, title=f"Training Accuracy, Leaky ReLU", xlabel="$\lambda$", ylabel="$\eta$", filename=f"cifar10_train_our.pdf")
heatmap(test_accuracy, xticks=lmbd_vals, yticks=eta_vals, title=f"Validation Accuracy, Leaky ReLU", xlabel="$\lambda$", ylabel="$\eta$", filename=f"cifar10_test_our.pdf")
