# Code

This code contains our code, and is split into three categories. The first consists of python codes containing different functions that are needed for our implementation, such as activation and cost functions. The second is the CNN-from-scratch implementation, which is made up of different classes that work together to make the CNN network. The third category consists of the runnable python scripts that are used to test the CNN network on datasets. More precisely, we here test our model on the MNIST[^1] and CIFAR-10[^2] datasets.

## Functions
- *funcs.py* contains the activation and cost functions, as well as a function for computing the correlations needed in the convolution layer.
- *scheduler.py* contains the different schedulers used for gradient descent when doing backpropagation.
- *plotting.py* contains functions for generating different types of plots, making sure that all figures are readable and consistently formatted.

## The CNN network
- *network.py* connects all the layer classes to build the full network.
- *layer.py* is an abstract class for the different layer classes.
- *convolution.py* is the class for the convolution layer.
- *maxpool.py* is the class for the maxpool layer.
- *averagepool.py* is the class for the averagepool layer.
- *flattenedlayer.py* is a class for flattenning output from convolution and pooling layers (which for each image-input are 3-dimensional arrays of height, width and depth) so that it can run through a regular fully connected neural network.
- *fullyconnected.py* is the class for the fully connected layers.

## The result generators
- HER MÅ VI LEGGE TIL NÅR VI VET HVILKE FILER SOM SKAL VÆRE MED! (bør gi navnet, hvilken data den ser på, og hvilken type figurer den produserer).
- *cifar10.py* is the file where we produce results for the CIFAR-10 dataset, the data is contained in the *cifar10data* folder. Figures that are produced are contained in the *Figures* folder.

[^1]: https://www.tensorflow.org/datasets/catalog/mnist
[^2]: https://www.cs.toronto.edu/~kriz/cifar.html
