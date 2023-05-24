from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras import optimizers
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from get_letter_finction import get_letter

data = pd.read_csv("C:/Users/Sohaila/Documents/هنا حيث الروعة كلها/sign language detection project/data sets/sign language data _arabic/sign_language_test.csv")
#print(data.head())
data = np.array(data)
print(data.shape)

np.random.shuffle(data)
train = data[0:50000].T
test =  data[50000:].T

train_data = (train[1:]).T
train_labels = train[0]
print(train_labels)
print(train_data.shape)

test_data   = test[1:].T
test_labels = test[0]
print(test_labels)
print(test_data.shape)

test_data  = test_data.reshape(len(test_data) , 64, 64)
train_data = train_data.reshape(len(train_data) , 64, 64)

test_data  = test_data.astype('float32')
train_data = train_data.astype('float32')
idx = random.randint(0,len(test_data))

#for one hot encoding we first need to convert the string values in the utput to numeric values
for label in range(len(train_labels)):
    if train_labels[label] == "ain":
        train_labels[label] = 0
    elif train_labels[label] == "al":
        train_labels[label] = 1
    elif train_labels[label] == "aleff":
        train_labels[label] = 2
    elif train_labels[label] == "bb":
        train_labels[label] = 3
    elif train_labels[label] == "dal":
        train_labels[label] = 4
    elif train_labels[label] == "dha":
        train_labels[label] = 5
    elif train_labels[label] == "dhad":
        train_labels[label] = 6
    elif train_labels[label] == "fa":
        train_labels[label] = 7
    elif train_labels[label] == "gaaf":
        train_labels[label] = 8
    elif train_labels[label] == "ghain":
        train_labels[label] = 9
    elif train_labels[label] == "ha":
        train_labels[label] = 10
    elif train_labels[label] == "haa":
        train_labels[label] = 11
    elif train_labels[label] == "jeem":
        train_labels[label] = 12
    elif train_labels[label] == "kaaf":
        train_labels[label] = 13
    elif train_labels[label] == "khaa":
        train_labels[label] = 14
    elif train_labels[label] == "la":
        train_labels[label] = 15
    elif train_labels[label] == "laam":
        train_labels[label] = 16
    elif train_labels[label] == "meem":
        train_labels[label] = 17
    elif train_labels[label] == "nun":
        train_labels[label] = 18
    elif train_labels[label] == "ra":
        train_labels[label] = 19
    elif train_labels[label] == "saad":
        train_labels[label] = 20
    elif train_labels[label] == "seen":
        train_labels[label] = 21
    elif train_labels[label] == "sheen":
        train_labels[label] = 22
    elif train_labels[label] == "ta":
        train_labels[label] = 23
    elif train_labels[label] == "taa":
        train_labels[label] = 24
    elif train_labels[label] == "thaa":
        train_labels[label] = 25
    elif train_labels[label] == "thal":
        train_labels[label] = 26
    elif train_labels[label] == "toot":
        train_labels[label] = 27
    elif train_labels[label] == "waw":
        train_labels[label] = 28
    elif train_labels[label] == "ya":
        train_labels[label] = 29
    elif train_labels[label] == "yaa":
        train_labels[label] = 30
    elif train_labels[label] == "zay":
        train_labels[label] = 31

for label in range(len(test_labels)):
    if test_labels[label] == "ain":
        test_labels[label] = 0
    elif test_labels[label] == "al":
        test_labels[label] = 1
    elif test_labels[label] == "aleff":
        test_labels[label] = 2
    elif test_labels[label] == "bb":
        test_labels[label] = 3
    elif test_labels[label] == "dal":
        test_labels[label] = 4
    elif test_labels[label] == "dha":
        test_labels[label] = 5
    elif test_labels[label] == "dhad":
        test_labels[label] = 6
    elif test_labels[label] == "fa":
        test_labels[label] = 7
    elif test_labels[label] == "gaaf":
        test_labels[label] = 8
    elif test_labels[label] == "ghain":
        test_labels[label] = 9
    elif test_labels[label] == "ha":
        test_labels[label] = 10
    elif test_labels[label] == "haa":
        test_labels[label] = 11
    elif test_labels[label] == "jeem":
        test_labels[label] = 12
    elif test_labels[label] == "kaaf":
        test_labels[label] = 13
    elif test_labels[label] == "khaa":
        test_labels[label] = 14
    elif test_labels[label] == "la":
        test_labels[label] = 15
    elif test_labels[label] == "laam":
        test_labels[label] = 16
    elif test_labels[label] == "meem":
        test_labels[label] = 17
    elif test_labels[label] == "nun":
        test_labels[label] = 18
    elif test_labels[label] == "ra":
        test_labels[label] = 19
    elif test_labels[label] == "saad":
        test_labels[label] = 20
    elif test_labels[label] == "seen":
        test_labels[label] = 21
    elif test_labels[label] == "sheen":
        test_labels[label] = 22
    elif test_labels[label] == "ta":
        test_labels[label] = 23
    elif test_labels[label] == "taa":
        test_labels[label] = 24
    elif test_labels[label] == "thaa":
        test_labels[label] = 25
    elif test_labels[label] == "thal":
        test_labels[label] = 26
    elif test_labels[label] == "toot":
        test_labels[label] = 27
    elif test_labels[label] == "waw":
        test_labels[label] = 28
    elif test_labels[label] == "ya":
        test_labels[label] = 29
    elif test_labels[label] == "yaa":
        test_labels[label] = 30
    elif test_labels[label] == "zay":
        test_labels[label] = 31


train_data = train_data/255
test_data  = test_data/255
print(train_data[1:])
print("**********")
print(test_data[1:])
print(len(train_data))


model = Sequential()
#first make the design of the model
#we will not make a stride to our convolution so the stride = 1 

#first layer of the model
model.add(Conv2D(32 , (2,2) , activation = 'relu' , input_shape = (64,64,1)))
model.add(MaxPooling2D((2,2)))

#second layer of the model
model.add(Conv2D(64 , (2,2) , activation = "relu"))
model.add(MaxPooling2D((2,2)))

#third layer of the model
model.add(Conv2D(128 , (2,2) , activation = "relu"))
model.add(MaxPooling2D((2,2)))

#fourth layer of the model
model.add(Flatten())
model.add(Dense(256 , activation = "relu"))

#fifth layer 
model.add(Dense(32 , activation = "softmax"))

# second step is to design the backward propagation algorithm
opt = optimizers.SGD(learning_rate = 0.01)
model.compile(loss = "categorical_crossentropy" , optimizer = "adam"  , metrics = ["accuracy"])

#now as the model is ready we can input our train data to it to train the data
#the epochs are the number of times our data will enter the model to improve the efficiencey of the prediction and the data is batched to decrease the time needed to train the model
#model.fit(train_data , train_labels , epochs = 5 , batch_size = 64)
model.fit(train_data , train_labels , batch_size = 128 , epochs = 8)

model.save("sign_model/amazing_model")

model.evaluate(test_data , test_labels)
























