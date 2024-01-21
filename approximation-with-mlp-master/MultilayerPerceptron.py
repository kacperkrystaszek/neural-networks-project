import numpy as np
from sklearn.metrics import mean_squared_error

class MultilayerPerceptron:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_layers = len(hidden_sizes) + 1
        self.activations = None
        self.layer_outputs = None

        # Initialize weights and biases for the network
        self.weights = [np.random.randn(input_size, hidden_sizes[0])]
        self.weights.extend([np.random.randn(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes) - 1)])
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size))

        self.biases = [np.zeros((1, size)) for size in hidden_sizes]
        self.biases.append(np.zeros((1, output_size)))

    # Activation function
    def bisigmoid(self, x):
        return np.tanh(x)    

    def d_bisigmoid(self, x):
        return 1 - self.bisigmoid(x)**2

    def forward(self, x):
        self.activations = [x]
        self.layer_outputs = []

        for i in range(self.num_layers):
            layer_input = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.layer_outputs.append(layer_input)
            activation = self.bisigmoid(layer_input)
            self.activations.append(activation)

        return self.activations[-1]

    def backward(self, x, y, output):
        errors = [None] * self.num_layers
        deltas = [None] * self.num_layers

        errors[-1] = y - output
        deltas[-1] = errors[-1] * self.d_bisigmoid(output)

        for i in reversed(range(self.num_layers - 1)):
            errors[i] = deltas[i + 1].dot(self.weights[i + 1].T)
            deltas[i] = errors[i] * self.d_bisigmoid(self.activations[i + 1])

        for i in range(self.num_layers):
            if i == 0:
                self.weights[i] += x.T.dot(deltas[i]) * self.learning_rate
            else:
                self.weights[i] += self.activations[i].T.dot(deltas[i]) * self.learning_rate

            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * self.learning_rate

    def train(self, x, y, epochs, *, validation_data):
        x_epochs = []
        mse_list = []
        
        for i in range(epochs):
            output = self.forward(x)
            self.backward(x, y, output)
            
            if not i % 500:
                _, y_test = validation_data
                # pred = self.predict(x_test)
                mse = mean_squared_error(y_test, output)
                x_epochs.append(i)
                mse_list.append(mse)
                print(f"Epoch {i}, MSE: {mse}")
            
        return x_epochs, mse_list

    def predict(self, x):
        return self.forward(x)