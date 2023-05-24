from tensorflow.keras.models import Sequential 
from tensorflow.keras.models import load_model 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras import optimizers
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
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

train_data = train_data/255
test_data  = test_data/255


#to get the name of the letter in the final stage after doing the prediction
def get_letter(num):
    if num == 0:
        c = "ع"    
    elif num == 1:
        c = "ال"
    elif num == 2:
        c = "أ"
    elif num == 3:
        c = "ب"
    elif num == 4:
        c = "د"
    elif num == 5:
        c = "ظ"
    elif num == 6:
        c = "ض"
    elif num == 7:
        c = "ف"
    elif num == 8:
        c = "ق"
    elif num == 9:
        c = "غ"
    elif num == 10:
        c = "ه"
    elif num == 11:
        c = "ح"
    elif num == 12:
        c = "ج"
    elif num == 13:
        c = "ك"
    elif num == 14:
        c = "خ"
    elif num == 15:
        c = "لا"
    elif num == 16:
        c = "ل"
    elif num == 17:
        c = "م"
    elif num == 18:
        c = "ن"
    elif num == 19:
        c = "ر"
    elif num == 20:
        c = "ص"
    elif num == 21:
        c = "س"
    elif num == 22:
        c = "ش"
    elif num == 23:
        c = "ط"
    elif num == 24:
        c = "ت"
    elif num == 25:
        c = "ث"
    elif num == 26:
        c = "ذ"
    elif num == 27:
        c = "ة"
    elif num == 28:
        c = "و"
    elif num == 29:
        c = "ئ"
    elif num == 30:
        c = "ي"
    elif num == 31:
        c = "ز"
        
    return c


def get_letter_test(label):
    if label == "ain":
        label = "ع"
    elif label == "al":
        label = "ال"
    elif label == "aleff":
        label = "أ"
    elif label == "bb":
        label = "ب"
    elif label == "dal":
        label = "د"
    elif label == "dha":
        label = "ظ"
    elif label == "dhad":
        label = "ض"
    elif label == "fa":
        label = "ف"
    elif label == "gaaf":
        label = "ق"
    elif label == "ghain":
        label = "غ"
    elif label == "ha":
        label =   "ه"
    elif label == "haa":
        label = "ح"
    elif label == "jeem":
        label = "ج"
    elif label == "kaaf":
        label = "ك"
    elif label == "khaa":
        label = "خ"
    elif label == "la":
        label = "لا"
    elif label == "laam":
        label = "ل"
    elif label == "meem":
        label = "م"
    elif label == "nun":
        label = "ن"
    elif label == "ra":
        label = "ر"
    elif label == "saad":
        label = "ص"
    elif label == "seen":
        label = "س"
    elif label == "sheen":
        label = "ش"
    elif label == "ta":
        label = "ط"
    elif label == "taa":
        label = "ت"
    elif label == "thaa":
        label = "ث"
    elif label == "thal":
        label = "ذ"
    elif label == "toot":
        label = "ة"
    elif label == "waw":
        label = "و"
    elif label == "ya":
        label = "ئ"
    elif label == "yaa":
        label = "ى"
    elif label == "zay":
        label = "ز"
    return label



model_path = "C:/Users/Sohaila/model.h5"


model = load_model(model_path)


idx = random.randint(0,len(test_data))
prediction = model.predict(test_data[idx , : ].reshape(1,64,64,1))


prediction = np.argmax(prediction)
prediction = get_letter(prediction)
real = test_labels[idx]
real = get_letter_test(real)

print("I think this image is for sign: ", prediction)
print("the picture is actually for: " ,real)

image = test_data[idx , :]*255
plt.gray()
plt.imshow(image)
plt.show()
