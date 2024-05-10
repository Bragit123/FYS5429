#We used Mortens lecture notes, with some tweeks, to see how the cnn of tensorflow could be implemented: https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/week44.html

from tensorflow.keras import datasets, layers
import numpy as np
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from network import Network
import time
from funcs import *
from scheduler import *
from plotting import * #Various plotting functions, we will use heatmap
from funcs import padding
from copy import copy

data_frac = 0.01
# data_frac = 0.001

digits = datasets.mnist.load_data(path="mnist.npz")
(x_train, y_train), (x_test, y_test) = digits #The data contains a test and a train set
x_train, x_test = x_train/255.0, x_test/255.0 #Normalising the pixel values to be in [0,1]
x_train = x_train[:][:][0:int(data_frac*len(x_train[:][:]))] #The data contains 60000 samples, 6000 should be enough for our purpose
x_test = x_test[:][:][0:int(data_frac*len(x_test[:][:]))]
y_train = y_train[0:int(data_frac*len(y_train))]
y_test = y_test[0:int(data_frac*len(y_test))]
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#Greyscale images should have depth 1
x_train = x_train[:,:,:,np.newaxis]
x_test = x_test[:,:,:,np.newaxis]


#Transforming the labels from a single digit to an array of length 10 with the digit corresponding to the index
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def create_convolutional_neural_network_keras(input_shape, receptive_field,
                                              n_filters, n_hidden_neurons, n_categories,
                                              eta, lmbd, activation):
    model = Sequential()
    model.add(layers.Conv2D(n_filters, (receptive_field, receptive_field), input_shape=input_shape, padding='valid',
              activation=activation, kernel_regularizer=regularizers.l2(lmbd)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(n_hidden_neurons, activation=activation, kernel_regularizer=regularizers.l2(lmbd)))
    model.add(layers.Dense(n_categories, activation='softmax', kernel_regularizer=regularizers.l2(lmbd)))

    sgd = optimizers.experimental.SGD(learning_rate=eta)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

def create_convolutional_neural_network_our_code(cost_func, input_shape, n_hidden_neurons, act_func, scheduler, n_filters):
    model = Network(cost_func, input_shape)
    model.add_Convolution_layer((n_filters, 3, 3, 1), act_func, copy(scheduler))
    model.add_MaxPool_layer(2, 2)
    model.add_Flattened_layer()
    model.add_FullyConnected_layer(n_hidden_neurons, act_func, copy(scheduler))
    model.add_FullyConnected_layer(10, softmax, scheduler)
    return model


epochs = 50
# batch_size = 400
batch_size = 60
batches = x_train.shape[0] // batch_size
input_shape = x_train.shape[1:4]
receptive_field = 3
n_filters = 5
n_hidden_neurons= 50
n_categories = 10

eta_vals = np.logspace(-2, 0, 3)
lmbd_vals = np.logspace(-5, -3, 3)

train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
# activation = "leaky_relu"
# act_func = LRELU
activation = "relu"
act_func = RELU

# for i, eta in enumerate(eta_vals):
#     for j, lmbd in enumerate(lmbd_vals):
#         CNN = create_convolutional_neural_network_keras(input_shape, receptive_field,
#                                                 n_filters, n_hidden_neurons, n_categories,
#                                                 eta, lmbd, activation)
#         CNN.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

#         train_accuracy[i][j] = CNN.evaluate(x_train, y_train)[1]
#         test_accuracy[i][j] = CNN.evaluate(x_test, y_test)[1]
#         print("Learning rate = ", eta)
#         print("Lambda = ", lmbd)
#         print(f"Test accuracy: {test_accuracy[i][j]:.3f}")
#         print()
# # Plotting the training and test accuracy
# heatmap(train_accuracy, xticks=lmbd_vals, yticks=eta_vals, title=f"Training Accuracy, sigmoid", xlabel="$\lambda$", ylabel="$\eta$", filename=f"../Figures/cnn_train_acc_tf.pdf")
# heatmap(test_accuracy, xticks=lmbd_vals, yticks=eta_vals, title=f"Test Accuracy, sigmoid", xlabel="$\lambda$", ylabel="$\eta$", filename=f"../Figures/cnn_test_acc_tf.pdf")

eta = 0.01
lmbd = 0.001
scheduler = Adam(eta, 0.9, 0.999)

cnn_tf = create_convolutional_neural_network_keras(input_shape, receptive_field,
                                                n_filters, n_hidden_neurons, n_categories,
                                                eta, lmbd, activation)
print("Training Tensorflow's network:")
t0 = time.time()
history = cnn_tf.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0)
t1 = time.time()
delta_time = t1-t0
print(f"  Time used: {delta_time:.4f}")

# print(history.history.keys())
val_accs_tf = history.history["val_accuracy"]
train_accs_tf = history.history["accuracy"]

epoch_arr = np.arange(epochs)
plt.figure()
plt.title("Tensorflow accuracies")
plt.plot(epoch_arr, val_accs_tf, label="Validation")
plt.plot(epoch_arr, train_accs_tf, label="Training")
plt.legend()
plt.savefig("tf_compare_accs.pdf")

cnn_our = create_convolutional_neural_network_our_code(CostLogReg, input_shape, n_hidden_neurons, act_func, scheduler, n_filters)
print("Training our network:")
t0 = time.time()
scores = cnn_our.train(x_train, y_train, x_test, y_test, epochs, batches, lmbd)
t1 = time.time()
delta_time = t1-t0
print(f"  Time used: {delta_time:.4f}")
val_accs_our = scores["val_accuracy"]
train_accs_our = scores["train_accuracy"]

plt.figure()
plt.title("Validation accuracies")
plt.plot(epoch_arr, val_accs_tf, label="Tensorflow")
plt.plot(epoch_arr, val_accs_our, label="Our network")
plt.legend()
plt.savefig("tf_compare_accs.pdf")

plt.figure()
plt.title("Training accuracies")
plt.plot(epoch_arr, train_accs_tf, label="Tensorflow")
plt.plot(epoch_arr, train_accs_our, label="Our network")
plt.legend()
plt.savefig("tf_compare_accs_train.pdf")

# train_accuracy_tf = CNN.evaluate(x_train, y_train)[1]
# test_accuracy_tf = CNN.evaluate(x_test, y_test)[1]