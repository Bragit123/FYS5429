import jax.numpy as jnp
from jax import random, vmap, grad

class Scheduler:
    """
    Abstract class for Schedulers
    """

    def __init__(self, eta):
        self.eta = eta

    # should be overwritten
    def update_change(self, gradient):
        raise NotImplementedError

    # overwritten if needed
    def reset(self):
        pass

class AdamMomentum(Scheduler):
    def __init__(self, eta, rho, rho2, momentum):
        super().__init__(eta)
        self.rho = rho
        self.rho2 = rho2
        self.momentum = momentum
        self.moment = 0
        self.second = 0
        self.n_epochs = 1
        self.change = 0


    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        self.moment = self.rho * self.moment + (1 - self.rho) * gradient
        self.second = self.rho2 * self.second + (1 - self.rho2) * gradient * gradient

        moment_corrected = self.moment / (1 - self.rho**self.n_epochs)
        second_corrected = self.second / (1 - self.rho2**self.n_epochs)

        self.change = self.change*self.momentum +self.eta * moment_corrected / (jnp.sqrt(second_corrected + delta))
        return self.change

    def reset(self):
        self.n_epochs += 1
        self.moment = 0
        self.second = 0

def derivate(func):
    if func.__name__ == "RELU":

        def func(X):
            return jnp.where(X > 0, 1, 0)

        return func

    elif func.__name__ == "LRELU":

        def func(X):
            delta = 10e-4
            return jnp.where(X > 0, 1, delta)

        return func

    else:
        return grad(func)


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

class FC:
    def __init__(
        self,
        input_length,
        output_length,
        act_func,
        seed=100
    ):
        self.input_length = input_length
        self.output_length = output_length
        self.weights_size = (self.input_length, self.output_length)
        self.bias_length = self.output_length
        self.act_func = act_func
        self.scheduler_weights = AdamMomentum(0.01, 0.9, 0.999, 0.01) #temporary
        self.scheduler_bias = AdamMomentum(0.01, 0.9, 0.999, 0.01) #temporary
        self.seed = seed

        self.X = None
        self.weights = None
        self.bias = None
        self.z = None
        self.a = None

        self.reset_weights()
    
    def reset_weights(self):
        rand_key = random.PRNGKey(self.seed)
        self.weights = random.normal(key=rand_key, shape=self.weights_size)
        self.bias = random.normal(key=rand_key, shape=(self.bias_length,))*0.01
    
    def reset_schedulers(self):
        self.scheduler_weights.reset()
        self.scheduler_bias.reset()
    
    def feed_forward(
        self,
        X
    ):
        self.X = X
        print(jnp.shape(X))
        print(jnp.shape(self.weights))
        print(jnp.shape(self.bias))
        self.z = X @ self.weights + self.bias
        self.a = self.act_func(self.z)

        return self.a
    
    def backpropagate(
        self,
        dC_doutput,
        lam = 1e-4
    ):
        X = self.X
        grad_act = vmap(vmap(derivate(self.act_func)))
        X_shape = jnp.shape(X)

        # ###########################
        # delta = dC_doutput * grad_act(self.z)
        # grad_weights = X.T @ delta/X_shape[0]
        # grad_biases = jnp.sum(delta, axis=0).reshape(1,jnp.shape(delta)[1])/X_shape[0]
        # grad_input = delta@self.weights.T
        # ###########################

        delta = grad_act(self.z) * dC_doutput

        grad_weights = X.T @ delta / X_shape[0]
        grad_bias = jnp.sum(delta, axis=0).reshape(1,jnp.shape(delta)[1]) / X_shape[0]
        grad_X = delta @ self.weights.T
        
        grad_weights += self.weights * lam

        self.weights -= self.scheduler_bias.update_change(grad_bias)
        self.bias -= self.scheduler_weights.update_change(grad_weights)

        return grad_X