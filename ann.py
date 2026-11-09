import numpy as np  # type: ignore -- numpy is installed and works, but mypy doesn't know that
import math

# Sigmoid activation
def sigmoid(X):
    return 1 / (1 + np.exp(-X))

# ReLU activation
def relu(X):
    return np.maximum(0, X)

# Log with offset
def safe_log(x):
    return math.log10(x + 1e-8)