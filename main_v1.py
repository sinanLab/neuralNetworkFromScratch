#!/usr/bin/env python3

import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1- x)

# mean squared error lose function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # initialize weights with random values
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)

        # initialize biases with zeros
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))


    def forward(self, X):
        # input to hidden layer
        self.hidden = sigmoid(np.dot(X, self.weights1) + self.bias1)

        # hidden layer to output
        self.output = sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output

    def backward(self, X, y, learning_rate):
        # calculate output error
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # calculate hidden layer error
        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        # update weights and biases
        self.weights2 += learning_rate * np.dot(self.hidden.T, output_delta)
        self.bias2 += learning_rate * np.sum(output_delta, axis = 0, keepdims = True)
        self.weights1 += learning_rate * np.dot(X.T, hidden_delta)
        self.bias1 += learning_rate * np.sum(hidden_delta, axis = 0, keepdims = True)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # forward pass
            output = self.forward(X)

            # backward pass and optimization
            self.backward(X, y, learning_rate)

            # print loss every 1000 epochs
            if epoch % 1000 ==0:
                loss = mse_loss(y, output)
                print(f"Epoch {epoch}, loss: {loss:.4f}")

def main(epochs = 20000, hidden_size=10, learning_rate = 0.05):
    # create a dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])
    y = np.array([[0], [1], [1], [0], [0.5], [0.9], [0.9]])

    # create neural network instance
    input_size = 2
    # hidden_size = 4 # you can experiment with this
    output_size = 1

    nn = NeuralNetwork(input_size, hidden_size, output_size)

    # train the network
    print("Training the neural network...")
    nn.train(X, y, epochs = epochs, learning_rate = learning_rate)
    
    # Test the network
    test_input = np.array([[0.3, 0.7], [0.7, 0.3], [0.1, 0.9], [0.9, 0.1], [0.4, 0.6]])
    predictions = nn.forward(test_input)
    print("\nTest prediction:")
    for i in range(len(test_input)):
        print(f"Input:{test_input[i]}, Predicted: {predictions[i][0]:.4f}")

if __name__ == "__main__":
    main()
