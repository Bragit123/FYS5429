#We used Mortens lecture notes, with some tweeks, to see how the cnn of tensorflow could be implemented: https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/week44.html

from tensorflow.keras import datasets, layers
import numpy as np
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from plotting import * #Various plotting functions, we will use heatmap

digits = datasets.mnist.load_data(path="mnist.npz")
(x_train, y_train), (x_test, y_test) = digits #The data contains a test and a train set
x_train, x_test = x_train/255.0, x_test/255.0 #Normalising the pixel values to be in [0,1]
x_train = x_train[:][:][0:int(0.1*len(x_train[:][:]))] #The data contains 60000 samples, 6000 should be enough for our purpose
x_test = x_test[:][:][0:int(0.1*len(x_test[:][:]))]
y_train = y_train[0:int(0.1*len(y_train))]
y_test = y_test[0:int(0.1*len(y_test))]
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
    model.add(layers.Conv2D(n_filters, (receptive_field, receptive_field), input_shape=input_shape, padding='same',
              activation=activation, kernel_regularizer=regularizers.l2(lmbd)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(n_hidden_neurons, activation=activation, kernel_regularizer=regularizers.l2(lmbd)))
    model.add(layers.Dense(n_categories, activation='softmax', kernel_regularizer=regularizers.l2(lmbd)))

    sgd = optimizers.experimental.SGD(learning_rate=eta)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

epochs = 50
batch_size = 400
input_shape = x_train.shape[1:4]
receptive_field = 3
n_filters = 10
n_hidden_neurons= 50
n_categories = 10

eta_vals = np.logspace(-2, 0, 3)
lmbd_vals = np.logspace(-5, -3, 3)

train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
activation = "sigmoid"
# for k in range(len(activation)):
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        CNN = create_convolutional_neural_network_keras(input_shape, receptive_field,
                                                n_filters, n_hidden_neurons, n_categories,
                                                eta, lmbd, activation)
        CNN.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        train_accuracy[i][j] = CNN.evaluate(x_train, y_train)[1]
        test_accuracy[i][j] = CNN.evaluate(x_test, y_test)[1]
        print("Learning rate = ", eta)
        print("Lambda = ", lmbd)
        print(f"Test accuracy: {test_accuracy[i][j]:.3f}")
        print()
#Plotting the training and test accuracy
heatmap(train_accuracy, xticks=lmbd_vals, yticks=eta_vals, title=f"Training Accuracy, sigmoid", xlabel="$\lambda$", ylabel="$\eta$", filename=f"../Figures/cnn_train_acc_tf.pdf")
heatmap(test_accuracy, xticks=lmbd_vals, yticks=eta_vals, title=f"Test Accuracy, sigmoid", xlabel="$\lambda$", ylabel="$\eta$", filename=f"../Figures/cnn_test_acc_tf.pdf")
