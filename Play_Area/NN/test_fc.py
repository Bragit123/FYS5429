import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, vmap
from fc import FC
from nn import NN

def sigmoid(X):
    try:
        return 1.0 / (1 + jnp.exp(-X))
    except FloatingPointError:
        return jnp.where(X > jnp.zeros(X.shape), jnp.ones(X.shape), jnp.zeros(X.shape))

def CostLogReg(target):

    def func(X):

        return -(1.0 / target.shape[0]) * jnp.sum(
            (target * jnp.log(X + 10e-6)) + ((1 - target) * jnp.log(1 - X + 10e-6))
        )

    return func



fc = FC(5, 2, sigmoid)
X = jnp.zeros((2,5))
X = X.at[0,:].set(jnp.linspace(1,5,5))
X = X.at[1,:].set(jnp.linspace(6,10,5))

a = fc.feed_forward(X)
t=jnp.array([[0],[1]])
grad_C = vmap(vmap(grad(CostLogReg(t))))
dC_doutput = grad_C(a)
dC_dX = fc.backpropagate(dC_doutput)

print(a)
print(dC_dX)


from sklearn import datasets
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

# Cost function
cost_func = CostLogReg
act_func = sigmoid

# Load dataset
cancer = datasets.load_breast_cancer()
X = cancer.data
X = minmax_scale(X, feature_range=(0,1), axis=0)
t = cancer.target
t = jnp.c_[t]
X_train, X_val, t_train, t_val = train_test_split(X, t, test_size=0.2)

# Layers
X_shape = jnp.shape(X)
t_shape = jnp.shape(t)
n_nodes_hidden = 100

input_layer = FC(X_shape[1], n_nodes_hidden, act_func)
# hidden_layer = FullyConnected(n_nodes_hidden, n_nodes_hidden, sigmoid)
output_layer = FC(n_nodes_hidden, t_shape[1], act_func)

# Create network
network = NN(cost_func)
network.add_layer(input_layer)
# network.add_layer(hidden_layer)
network.add_layer(output_layer)

epochs = 100
scores = network.train(X_train, t_train, X_val, t_val, epochs=epochs)

epoch_arr = jnp.arange(1, 100, 1)

plt.plot(epoch_arr, scores["train_error"], label="Train")
plt.plot(epoch_arr, scores["val_error"], label="Val")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend()
plt.savefig("test.pdf")