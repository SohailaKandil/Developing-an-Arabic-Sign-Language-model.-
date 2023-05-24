from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import numpy as np
import pandas as pd

data = pd.read_csv("sign_language_test.csv")
print(data.head())

data = np.array(data)
np.random.shuffle(data)
train = data[0:2500].T
test =  data[2500:].T

train_data = train[1:]/255
train_labels = train[0]

test_data = test[1:]
test_labels = test[0]
print(test_labels)
print(train_data.shape)

