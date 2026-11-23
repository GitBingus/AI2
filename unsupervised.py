import numpy as np # type: ignore

# -----------------------------
# Dataset (2 features)
# -----------------------------
X = np.array ([
    [5,5,7,2],
    [3,4,6,3],
    [1,2,5,5],
    [4,5,6,2],
    [2,3,6,4],
    [3,3,5,3],
    [6,5,8,1],
    [5,4,7,2],
    [2,2,5,4],
    [4,5,6,2]
])

Y = np.array([
    [1],[1],[0],[1],[0],[0],[1],[1],[0],[1]
])

input_nodes = X.shape[1] # 4 features
hidden_nodes = 4
hidden_nodes2 = 3
output_nodes = 1

# -----------------------------
# Initialize weights and biases
# -----------------------------
np.random.seed(42)
W1 = np.random.uniform(-0.5, 0.5, (input_nodes, hidden_nodes))
b1 = np.random.uniform(-0.5, 0.5, (1, hidden_nodes))

W2 = np.random.uniform(-0.5, 0.5, (hidden_nodes, hidden_nodes2))
b2 = np.random.uniform(-0.5, 0.5, (1, hidden_nodes2))

W3 = np.random.uniform(-0.5, 0.5, (hidden_nodes2, output_nodes))
b3 = np.random.uniform(-0.5, 0.5, (1, output_nodes))

# -----------------------------
# Activation functions
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)  # x is already sigmoid output

# -----------------------------
# Forward pass
# -----------------------------
def forward(X_row, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X_row, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# -----------------------------
# SGD Training (1 hidden layer, 3 nodes)
# -----------------------------
lr = 0.1
epochs = 100

for epoch in range(epochs):
    for i in range(len(X)):
        X_row = X[i].reshape(1,-1)
        Y_row = Y[i].reshape(1,-1)

        # Forward
        _, A1, _, A2, _, A3 = forward(X_row, W1, b1, W2, b2, W3, b3)

        # Compute deltas
        delta3 = A3 - Y_row
        delta2 = np.dot(delta3, W3.T) * sigmoid_derivative(A2)
        delta1 = np.dot(delta2, W2.T) * sigmoid_derivative(A1)

        # Update weights immediately (SGD)
        W3 -= lr * np.dot(A2.T, delta3)
        b3 -= lr * delta3
        W2 -= lr * np.dot(A1.T, delta2)
        b2 -= lr * delta2
        W1 -= lr * np.dot(X_row.T, delta1)
        b1 -= lr * delta1

# -----------------------------
# Test a sample
# -----------------------------
row6 = X[5].reshape(1,-1)

_, _, _, _, _, pred = forward(row6, W1, b1, W2, b2, W3, b3)
print("Pass probability for row 6:", pred[0,0])
