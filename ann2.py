import numpy as np # type: ignore

class ANN2:
    def __init__(self) -> None:
        self.X = np.array([[5, 2], [3, 4], [6, 1], [2, 5], [4, 3], [5, 3], [7, 1], [3, 6], [4, 2], [6, 2]])
        self.Y = np.array([[1], [0], [1], [0], [0], [1], [1], [0], [1], [1]])

        np.random.seed(42)
        self.W1 = np.random.uniform(-0.5, 0.5, (2,3)) # Input -> Hidden
        self.b1 = np.random.uniform(-0.5, 0.5, (1,3)) # Hidden bias
        self.W2 = np.random.uniform(-0.5, 0.5, (3,1)) # Hidden -> Output
        self.b2 = np.random.uniform(-0.5, 0.5, (1,1)) # Output bias

        row6 = self.X[5].reshape(1,-1)
        _, _, _, A2_6_before = self.forward(row6, self.W1, self.b1, self.W2, self.b2)

        print("Step 4: Row 6 output before training (Pass probability):", A2_6_before)

        lr = 0.1
        row1 = self.X[0].reshape(1,-1)
        target1 = self.Y[0].reshape(1,-1)

        # Forward pass for row 1
        Z1, A1, Z2, A2 = self.forward(row1, self.W1, self.b1, self.W2, self.b2)
        delta2 = A2 - target1

        # Gradients for hidden -> output
        dW2 = np.dot(A1.T, delta2)
        db2 = delta2

        # hidden layer error
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(A1)

        # Gradients for input -> hidden
        dW1 = np.dot(row1.T, delta1)
        db1 = delta1

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        _, _, _, A2_6_after = self.forward(row6, self.W1, self.b1, self.W2, self.b2)
        print("Step 6: Row 6 output after one training step (Pass probability):", A2_6_after)

        print("Step 6: Row 6 final output (Pass, Fail):", np.hstack([A2_6_after, 1-A2_6_after]))

        print("\nStep 7: Is the network improving?")
        print("Pass probability before:", A2_6_before[0,0])
        print("Pass probability after:", A2_6_after[0,0])
        if abs(A2_6_after - target1[0,0]) < abs(A2_6_before - target1[0,0]):
            print("→ The ANN prediction is getting closer to the correct output. Learning is happening!")
        else:
            print("→ One step may not be enough; the network may not show improvement yet.")


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x) # x is already sigmoid output

    def forward(self, X_ROW, W1, b1, W2, b2):
        z1 = np.dot(X_ROW, W1) + b1
        a1 = self.sigmoid(z1)

        z2 = np.dot(a1, W2) + b2
        a2 = self.sigmoid(z2)

        return z1, a1, z2, a2


if __name__ == "__main__":
    ann = ANN2()