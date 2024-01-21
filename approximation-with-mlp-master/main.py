import random
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from MultilayerPerceptron import MultilayerPerceptron

# Create training data by recreating function
def create_func(x_values, y_values):
    result_x = []
    result_y = []
    
    for index, x in enumerate(x_values):
        if index + 1 != len(x_values):
            result_x.append(np.linspace(x-1, x, 11)[:-1])
        else:
            result_x.append(np.linspace(x-1, x, 11))
        
    for index, y in enumerate(y_values[1:], start=1):
        if index + 1 != len(y_values):
            result_y.append(np.linspace(y_values[index-1], y, 11)[:-1])
        else:
            result_y.append(np.linspace(y_values[index-1], y, 11))
    
    return np.concatenate(result_x).reshape(-1, 1), np.concatenate(result_y).reshape(-1, 1)

# Test data
x_train = np.array([float(x) for x in range(11)]).reshape(-1, 1)
y_train = np.array([1.0, 1.32, 1.6, 1.54, 1.41, 1.01, 0.6, 0.42, 0.2, 0.51, 0.8]).reshape(-1, 1)

x, y = create_func(x_train[1:], y_train)

# x_train = np.setdiff1d(x, x_test).reshape(-1, 1)
# y_train = np.insert(np.setdiff1d(y, y_test, assume_unique=True), 58, 0.51).reshape(-1, 1) # There is duplicate of '0.51' value which has to be inserted manually

# Normalize the data between 0 and 1
y_max = y_train.max()
y_train /= y_max

for _ in range(100):
    p = 16
    hidden_sizes = [16, 16, 16, 16]
    # print(f"hidden_sizes={hidden_sizes}")
    # Create an instance of the MLP with 3 hidden layers with 100 neurons each
    model = MultilayerPerceptron(input_size=1, hidden_sizes=hidden_sizes, output_size=1, learning_rate=0.001)

    # Train the model
    x_epochs, mse_list = model.train(x_train, y_train, 10000, validation_data=(x_train, y_train))

    # Test the model on new data
    predictions = model.predict(x)
    # print(f"Final MSE: {mean_squared_error(y_train, predictions[::10])}")

    # MSE for every 500 epochs
    plt.figure(figsize=(8, 6))
    plt.plot(x_epochs, mse_list, color='b', label='MSE')
    plt.title('MSE(epoch)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(x_train, y_train * y_max, 'bo', label='Training data')
    plt.plot(x, predictions * y_max, "r+", label='Predictions')
    plt.title(f'Function approximation by multilayer perceptron with {len(hidden_sizes)} hidden layers with {p} neurons each')
    plt.xlabel('t[h]')
    plt.ylabel('h(t)[m]')
    plt.ylim(bottom=0, top=2)
    plt.legend()
    plt.show()
    time.sleep(3)
    plt.close('all')