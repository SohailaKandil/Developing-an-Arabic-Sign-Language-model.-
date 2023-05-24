import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from get_letter_number import get_num
from get_number_letter import get_letter

# function to read the data at the beginning to understand it giving it the path of your data on your device
data = pd.read_csv("C:/Users/Sohaila/Documents/هنا حيث الروعة كلها/sign language detection project/data sets/sign language data _arabic/sign_language_test.csv")
#print(data.head())

# convert the pandas DataFrame to a numpy array and shuffle it
data = np.array(data)
np.random.shuffle(data)

# get the shape of the data array
m, n = data.shape
print(m,n)

# use the first 1000 samples as a development set
data_dev = data[0:1000].T
labels = data_dev[0]
pixels = data_dev[1:n] / 255



# use the rest of the samples as the training set
train = data[1000:m].T
labels_train = train[0]
pixels_train = train[1:n] / 255
# why this line: pixels_train = pixels_train / 250: because we want the weight to have a segnificant effect on the prediction.

# define the activation function relu
def relu(z):
    return np.maximum(z, 0)

# define the softmax activation function
def soft_max(z):
    return np.exp(z) / sum(np.exp(z))

# define the derivative of relu function
def der_relu(z):
    return z > 0

# initialize the parameters of the neural network
#w1 are the weights for every combination from the input layer to the first hidden layer , w2 is the array oe wights from hidden layer1 to #hidden layer2 , b1 are the biases of the first hidden layer, and b2 is the biases of the second hidden layer
def init_params():
    w1 = np.random.rand(32 , 4096) - 0.5  # why not randn(10,784) why should the values be between -0.5 and 0.5
    b1 = np.random.rand(32 , 1) - 0.5 
    w2 = np.random.rand(32 , 32) - 0.5
    b2 = np.random.rand(32, 1) - 0.5
    return w1, b1, w2, b2

# implement the forward propagation step
#z1 is the array of the first hidden layer, a1 is the values of the nodes in the first hidden layer after applying the activation function #on it,and the same is the case with layer 2.
def forward_propagation(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = soft_max(z2)
    return z1, a1, z2, a2

#oe hot encoding of the output
def one_hot_encoding (arr): 
    one_hot = np.zeros((arr.size, 32))
    for num , letter in enumerate(arr):
        c = get_num (letter)
        one_hot [num][c] = 1
        one_hot = one_hot.T
        return one_hot
    
    
# implement the back propagation step
def backward_propagation(z1, a1, z2, a2, w2, x, out):   # should the input has the w1 we actually didn't use it
    m = out.size
    one_hot = one_hot_encoding(out)
    dz2 = a2 - one_hot   # why not multiplied by 2 and shouldn's this be da2 not dz2
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)   # why 1 / m * np.sum(dz2,1) raises an error and shouldn't it be the correct value
    dz1 = w2.T.dot(dz2) * (der_relu(z1))
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1)    # also why 1 / m * np.sum(dz1,1) raises an error and shouldn't it be the correct value
    return dw1, db1, dw2, db2


def update_params(dw1, db1, dw2, db2, w1, b1, w2, b2, alpha):
    # Update the parameters w1, b1, w2, b2 using their corresponding gradients and learning rate alpha
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def get_predictions(z):
    # Return the index of the maximum value in each column of z
    return np.argmax(z, 0)

def find_accuracy(prediction, real):
    # Get the predicted labels and compare them to the true labels to compute accuracy
    prediction = get_predictions(prediction)
    prediction = get_letter (prediction)
    print("I think it is:" , "\n")
    print(prediction, "\n")
    print("while they are actually:")
    print(real)
    return np.sum((prediction == real)) / real.size

def design_network(inp, out, alpha, iterations):
    # Initialize the weights and biases
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        # Perform forward and backward propagation to get gradients and update weights and biases
        z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, inp)
        dw1, db1, dw2, db2 = backward_propagation(z1, a1, z2, a2, w2, inp, out)
        w1, b1, w2, b2 = update_params(dw1, db1, dw2, db2, w1, b1, w2, b2, alpha)
        if i % 10 == 0:
            # Print the accuracy every 10 iterations
            print("at iteration:", i)
            print("the accuracy is:", find_accuracy(a2, out))
            print("\n", "*************")
    return w1, b1, w2, b2



def predict (w1, b1, w2, b2, inp):
    z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, inp)
    prediction = get_predictions (a2)
    return prediction
    
def guess_sign(index , w1, b1, w2, b2):
    image = pixels[: , index , None]
    prediction = predict(w1, b1, w2, b2, image)
    real_image = labels[index]
    
    prediction = get_letter(prediction)    
    print("the prediction is: " , prediction)
    print("the real value is: " , real_image)
    
    show_image = image.reshape((64,64)) * 255
    
    plt.gray()
    plt.imshow(show_image)
    plt.show()





w1, b1, w2, b2 = design_network (pixels_train , labels_train , 0.1 , 500)
guess_sign(4 , w1, b1, w2, b2)
guess_sign(7 , w1, b1, w2, b2)
guess_sign(8 , w1, b1, w2, b2)



