import numpy as np
from fullyconnected import FullyConnected
from funcs import sigmoid, softmax

class SoftmaxLayer(FullyConnected):
    def __init__(self, input_length: int, output_length: int, scheduler, seed: int = 100):
        super().__init__(input_length, output_length, sigmoid, scheduler, seed)
        self.sigmoid_output = None
    
    def feed_forward(self, input: np.ndarray):
        """
        Feeds input forward through the neural network.

        ## Parameters:
            - input (ndarray): Two-dimensional input array to be fed forward through
            the neural network. The first axis corresponds to the different
            inputs, and the second axis corresponds to the input values.

        ## Returns:
            ndarray: Two-dimensional array containing the output from the layer.
            The axes correspond to the same axes as the input.
        """
        self.input = input # Save input for use in backpropagate().
        #num_inputs = np.shape(input)[0]
        self.z = input @ self.weights + self.bias

        # calculate a, add bias
        self.sigmoid_output = sigmoid(self.z)
        output = softmax(self.z)

        return output

    def backpropagate(self, dC_doutput: np.ndarray, lmbd: float = 0.01):
        """
        Backpropagates through the layer to find the partial derivatives of the
        cost function with respect to each weight, bias and input value. The
        derivatives with respect to weights and biases are used to update the
        weights and biases using gradient descent, while the derivatives with
        respect to the input is returned for use in backpropagation through
        previous layers.

        ## Parameters
            - dC_doutput (ndarray): Two-dimensional array containing the
              partial derivatives of the cost function with respect to every
              output value from this layer. The first axis is the different
              inputs, and the second axis corresponds to every partial
              derivative.
            - lmbd (float): WILL BE CHANGED TO SCHEDULER.

        ## Returns
            ndarray: Partial derivatives of the cost function with respect to
            every input value to this layer.
        """
        

        #dC_da = dC_doutput * grad_act(self.z)
        delta_matrix = dC_doutput * grad_act(self.z)
        grad_weights = input.T @ delta_matrix/input_size[0]
        grad_biases = np.sum(delta_matrix, axis=0).reshape(1, np.shape(delta_matrix)[1])/input_size[0]
        grad_input = delta_matrix @ self.weights.T
        #grad_input = self.weights.T@delta_matrix

        # print(f"Before : {grad_weights}")
        grad_weights = grad_weights + self.weights * lmbd
        # print(f"After : {grad_weights}")

        #for i in range(self.input_length):
            #for j in range(self.output_length):
                ## Compute the gradients.
            #    grad_weights = grad_weights.at[i,j].set(np.sum(dC_doutput[:,j] * grad_act(self.z[:,j]) * input[:,i])/input_size[0])
            #    grad_biases = grad_biases.at[j].set(np.sum(1*grad_act(self.z[:,j])*dC_doutput[:,j])/input_size[0])

            #    zero_matrix = np.zeros(input_size)
            #    grad_input += zero_matrix.at[:,i].set(dC_doutput[:,j] * grad_act(self.z[:,j]) * self.weights[i,j])


        # scheduler_weights = Adam(0.001, 0.9, 0.999)

        # self.scheduler_bias.reset()
        # self.scheduler_weights.reset()
        self.weights -= self.scheduler_weights.update_change(grad_weights)
        # scheduler_bias = Adam(0.001, 0.9, 0.999)
        update_bias = self.scheduler_bias.update_change(grad_biases)
        self.bias -= update_bias

        return grad_input
