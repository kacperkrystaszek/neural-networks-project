import numpy as np
import requests, gzip, os, hashlib
import matplotlib.pyplot as plt
from PIL import Image

#fetch data
path='./data'
def fetch(url):
    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

#Train set from MNIST - random split for Train and Validation sets
X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]

rand = np.arange(60000)
np.random.shuffle(rand)

train_no = rand[:50000]
val_no = np.setdiff1d(rand, train_no)

X_train, X_val = X[train_no, :, :], X[val_no, :, :]
Y_train, Y_val = Y[train_no], Y[val_no]

#Test set from MNIST
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

#Neural network
#Input Layer - 784 units - every pixel of input image 28x28 size
#Hidden Layer - 128 units
#Output Layetr - 10 units - predicted digit

def init_layer(x, y):
    layer = np.random.uniform(-1., 1., size=(x,y))/np.sqrt(x*y)
    return layer.astype(np.float32)

np.random.seed(42)
l1 = init_layer(28*28, 128)
l2 = init_layer(128, 10)

#Sigmoid Function - used to change values from input layer to values between 0 and 1
def sigmoid(x):
    return 1/(np.exp(-x)+1)

def d_sigmoid(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

#Softmax Function - create probabilities, normalize results of output vector
# def softmax(x):
#     exps = np.exp(x)
#     return exps/np.sum(exps)
def softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)

# def d_softmax(x):
#     exps = np.exp(x)
#     return exps/np.sum(exps)*(1-exps/np.sum(exps))

def d_softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))

# Activation functions tests - can be deleted later
# l2_sample_output = np.array([12, 34, -67, 23, 0, 134, 76, 24, 78, -98])
# l2_normalized = softmax(l2_sample_output)
# max_l2_val = np.argmax(l2_normalized)
# print(max_l2_val, l2_sample_output[max_l2_val])

#Forward/Backward Pass algorithm
def forward_backward_pass(x, y):
    #Convert correct digit from Y_train to 10 elements vector to match model output
    targets = np.zeros((len(y), 10), np.float32)
    targets[range(targets.shape[0]), y] = 1

    #Forward pass
    #Multiply x (1,784) with L1 (784,128)
    #Use activation function to get sigmoid (1, 128)
    #Multiply sigmoid on L2 (128, 10)
    #Use softmax to get output layer (1,10)

    #L1 -> L2
    x_l1 = x.dot(l1)
    x_sigmoid = sigmoid(x_l1)

    #L2 -> L3
    x_l2 = x_sigmoid.dot(l2)
    out = softmax(x_l2)

    #Backward pass
    #Calculate error - how far are we from ideal answer
    #Use softmax derivative to get direction of error
    #Update L2
    #Calculate error of l1 with derivative of sigmoid to get direction
    #Update L1 by the error

    error = 2*(out-targets)/out.shape[0]*d_softmax(x_l2)
    update_l2 = x_sigmoid.T@error

    error =((l2).dot(error.T)).T*d_sigmoid(x_l1)
    update_l1 = x.T@error

    return out, update_l1, update_l2

#Training parameters
epochs = 10000
learning_rate = 0.001
batch = 128

accuracies = []
losses = []
val_accuracies = []

for i in range(epochs):
    #Get learning sample
    sample = np.random.randint(0, X_train.shape[0], size=(batch))
    x = X_train[sample].reshape((-1, 28*28))
    y = Y_train[sample]

    #Apply forward and backward pass to learn
    out, update_l1, update_l2 = forward_backward_pass(x, y)

    #Get the most probable value, find its accuracy
    category = np.argmax(out, axis=1)
    accuracy = (category==y).mean()
    accuracies.append(accuracy)

    #Calculate MSE
    loss = ((category-y)**2).mean()
    losses.append(loss.item())

    #Update Layers including error from training sample
    l1 = l1 - learning_rate*update_l1
    l2 = l2 - learning_rate*update_l2

    #Validate every 20 epochs:
    if (i%20==0):
        X_val = X_val.reshape((-1, 28*28))
        val_out = np.argmax(softmax(sigmoid(X_val.dot(l1)).dot(l2)), axis=1)
        val_acc = (val_out==Y_val).mean()
        val_accuracies.append(val_acc.item())
    #Display training process every 500 epochs:
    if (i%500==0):
        print(f'Training progress: {i} epoch: train accuracy: {accuracy:.3f} | validation accuracy: {val_acc: .3f}')

#Validation accuracy
plt.ylim(-0.1, 1.1)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.title("Model accuracy")
plt.plot(accuracies)
plt.show()

plt.title("Model validation accuracy")
plt.ylabel("Validation iteration")
plt.xlabel("Accuracy")
plt.plot(val_accuracies)
plt.show()

#Testing set
test_out = np.argmax(softmax(sigmoid(X_test.dot(l1)).dot(l2)), axis=1)
test_acc = (test_out==Y_test).mean().item()
print(f'Model accuracy = {test_acc*100:.2f}%')

#Test by hand
input_number = [[0,0,0,0,0,0,0],
                [0,0,0,10,0,0,0],
                [0,0,0,10,0,0,0],
                [0,0,0,10,0,0,0],
                [0,0,0,10,0,0,0],
                [0,0,0,10,0,0,0],
                [0,0,0,0,0,0,0]]

input_number = np.concatenate([np.concatenate([[x]*4 for x in y]*4) for y in input_number])
input_number = input_number.reshape(1, -1)
plt.imshow(input_number.reshape(28,28))
plt.show()

x = np.argmax(sigmoid(input_number.dot(l1)).dot(l2), axis=1)
print('Test results:')
print(sigmoid(input_number.dot(l1)).dot(l2))
print(f'Number predicted: {x}')

#Test by handmade image
image = Image.open('image.jpg')
image = np.asarray(image)
data = np.zeros((28, 28), np.float32)
for i in range(28):
    for j in range(28):
        data[i][j] = image[i][j].mean()

plt.imshow(data)
plt.show()

x2 = np.argmax(sigmoid(data.reshape(1, -1).dot(l1)).dot(l2), axis=1)
print('Test results:')
print(sigmoid(data.reshape(1, -1).dot(l1)).dot(l2))
print(f'Number predicted: {x2}')
