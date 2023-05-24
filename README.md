# Developing-an-Arabic-Sign-Language-model.-
In this project, I created an Artificial Neural Network (ANN) from scratch to predict digits. The training data consisted of 42000
images of handwritten digits, and the model achieved an accuracy of approximately 85%.
Later, I developed a Convolutional Neural Network (CNN), which is better suited for image training, to build a sign language
model. This model was trained using Keras and TensorFlow, using a dataset of 50,000 images representing 32 different Arabic sign
letters and it was able to predict the sign language images with accuracy of 95.9%.

## 1. Building a digit recognition neural network from scratch
### 1.1. Digits dataset description
Description: In this project, we built a neural network to pre-
dict digits represented in gray scale images. The dataset con-
sists of 42,000 images stored in an Excel sheet. Each row rep-
resents the pixel values of an image, with a size of 28 by 28
pixels (784 total pixels).
### 1.2. Project description
Building the Artificial Neural
Network (ANN) is done through these phases, as follows:
1. Divide the available data into a training set and a testing
set.
2. Rescale the pixel values of the dataset by dividing each
pixel value by 255. This enhances the significance of
weights in predicting the images and improves accuracy.
3. Initialize the weights and biases with random values.
4. Perform the forward propagation function. In this step, the
network predicts the inputs and calculates the loss function
(error).
5. Execute the backward propagation function to adjust the
weights and biases. This returns the updated weights and
biases that reduce the error.
6. Repeat steps 4 and 5, where each forward propagation uses
the weights and biases obtained from the previous back-
ward propagation. Measure accuracy by comparing the
predicted output with the true values. Stop repeating when
a satisfactory accuracy is achieved.
7. Evaluate the network’s performance using the test dataset.

