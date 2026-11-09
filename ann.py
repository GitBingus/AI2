import numpy as np  # type: ignore -- numpy is installed and works, but mypy doesn't know that
import math

w1 = np.array([[0.1, 0.2], [0.4, 0.3], [0.2, 0.5]])
b1 = np.array([[0.1], [0.2], [0.3]])

w2 = np.array([[0.3, 0.6, 0.9]])
b2 = np.array([[0.1]])

print(f'{w1=}, {b1=}, {w2=}, {b2=}')

# Sigmoid activation
def sigmoid(X):
    return 1 / (1 + np.exp(-X))

# ReLU activation
def relu(X):
    return np.maximum(0, X)

# Log with offset
def safe_log(x):
    return math.log(x + 1e-8)

X = np.array([[5, 2], [1, 3]])
y = np.array([[1], [0]])

print(f'{X.T=}')

sigmoid_output1 = sigmoid(np.dot(w1, X.T) + b1)
sigmoid_output2 = sigmoid(np.dot(w2, sigmoid_output1) + b2)

cost = -(1 / 2) * np.sum(y * safe_log(sigmoid_output2[0][0]) + (1 - y) * safe_log(1 - sigmoid_output2[0][1]))

print(f'Predictions: {sigmoid_output2[0][0]}, {sigmoid_output2[0][1]}')
print(f'Cost: {cost}')