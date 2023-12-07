import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

x_values = np.linspace(-5, 5, 100)
fig, ax = plt.subplots()
ax.plot(x_values, sigmoid(x_values), c='darkorange')
ax.set_xlim(-5, 5)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel('Neuron Input')
ax.set_ylabel('Neuron Activation')
fig.savefig('activation_functions/sigmoid.png', dpi=300, bbox_inches='tight')

fig, ax = plt.subplots()
ax.plot(x_values, relu(x_values), c='darkorange')
ax.set_xlim(-5, 5)
ax.set_ylim(-0.1, 5)
ax.set_xlabel('Neuron Input')
ax.set_ylabel('Neuron Activation')
fig.savefig('activation_functions/relu.png', dpi=300, bbox_inches='tight')
