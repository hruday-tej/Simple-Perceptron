import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidDerivative(x):
    return x * (1-x)

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])


training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_wights = 2 * np.random.random((3,1))-1

print('Random Starting Synaptic Wreights are:')
print(synaptic_wights)

for iteration in range(20000):
    
    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_wights))

    error = training_outputs - outputs
    adjustments = error * sigmoidDerivative(outputs)

    synaptic_wights += np.dot(input_layer.T, adjustments)

print('Synaptic weights after training:')
print(synaptic_wights)

print('OUTPUTS AFTER TRAINING: ')
print(outputs)
