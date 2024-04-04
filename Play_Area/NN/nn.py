import jax.numpy as jnp
from jax import vmap, grad
from sklearn.utils import resample

class NN:
    def __init__(
        self,
        cost_func,
    ):
        self.cost_func = cost_func
        self.layers = []
        self.num_layers = 0
    
    def add_layer(self, layer):
        self.layers.append(layer)
        self.num_layers += 1
    
    def reset_weights(self):
        for layer in self.layers:
            layer.reset_weights()
    
    def reset_schedulers(self):
        for layer in self.layers:
            layer.reset_schedulers()
    
    def feed_forward(self, X):
        a = self.layers[0].feed_forward(X)
        for i in range(1, self.num_layers):
            a = self.layers[i].feed_forward(a)
        
        return a
    
    def backpropagate(self, output, target):
        grad_cost = vmap(vmap(self.cost_func(target)))
        dC_doutput = grad_cost(output)

        for i in range(self.num_layers-1, -1, -1):
            dC_doutput = self.layers[i].backpropagate(dC_doutput)
    
    def train(self, X_train, t_train, X_val = None, t_val = None, epochs=100, batches=1):
        self.reset_weights()
        batch_size = X_train.shape[0] // batches

        train_cost = self.cost_func(t_train)
        train_err = jnp.zeros(epochs)
        train_acc = jnp.zeros(epochs)

        if X_val is not None:
            val_cost = self.cost_func(t_val)
            val_err = jnp.zeros(epochs)
            val_acc = jnp.zeros(epochs)
        
        X_train, t_train = resample(X_train, t_train, replace=False)

        for e in range(epochs):
            print("Epoch: " + str(e+1) + "/" + str(epochs))
            # for b in range(batches):
            #     if b == batches - 1:
            #         X_batch = X_train[b*batch_size :]
            #         t_batch = t_train[b * batch_size :]
            #     else:
            #         X_batch = X_train[b*batch_size : (b+1)*batch_size]
            #         t_batch = t_train[b*batch_size : (b+1)*batch_size]
                
            #     output_batch = self.feed_forward(X_batch)
            #     self.backpropagate(output_batch, t_batch)
            
            self.reset_schedulers()

            X_batch = X_train
            t_batch = t_train
            output_batch = self.feed_forward(X_batch)
            self.backpropagate(output_batch, t_batch)

            output_train = self.feed_forward(X_train)
            train_err = train_err.at[e].set(train_cost(output_train))
            if X_val is not None:
                output_val = self.feed_forward(X_val)
                val_err = val_err.at[e].set(val_cost(output_val))
            
        scores = {
            "train_error": train_err,
            "val_error": val_err
        }

        return scores