# Developing-an-Arabic-Sign-Language-model-using-keras-tensorflow-library-and-a-digit-recognition-model-from-scratch.
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
[Most important reference used for this project](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook)

## 2. Building an arabic sign language model using keras, Tensorflow library
### 2.1. Arabic sign language dataset description
For this project, we utilized a dataset containing
53,392 images representing the 32 letters of the Arabic sign
language (ArSL). Each image has dimensions of 64 by 64 pix-
els, resulting in a total of 4,096 pixels per image. All pixel
values range from 0 for black and 255 for white. The dataset is
composed of several JPG images. [click this link to access the dataset]([https://www.openai.com](https://www.data-in-brief.com/article/S2352-3409(19)30128-3/fulltext))

### 2.2. Project Description: Sign Language Recognition and Interpretation
The model training and construction involve the following
steps:
1. Data Preparation: The images are converted into pixel val-
ues and stored in an Excel sheet. Each row in the sheet
represents the pixel values of a single image.
2. Train-Test Split: The data is divided into training images
(50,000) and testing images (remaining).
3. Shuffling: The data is shuffled to enhance the neural net-
work’s performance and eliminate potential biases.
4. Reshaping: The pixel values of each image are reshaped
into a 64 by 64 format. This is necessary for convolutional
operations using kernels, as the input to the CNN must
have the same dimensions as the original image.
5. Label and Pixel Arrays: The training and testing datasets
are separated into label arrays and pixel arrays. The index
i of the labels array corresponds to the label of the pixel
array at index i.
6. One-Hot Encoding: The labels are assigned unique num-
bers, and one-hot encoding is applied to transform these
numbers into binary vectors.
7. Rescaling the pixel values of the dataset: To enhance the
significance of weights in image prediction and improve
overall accuracy, we rescale each pixel value by dividing
it by 255.
8. Designing the model: In this step, we
construct the layers of the model. Firstly, we add a convo-
lutional layer with 32 nodes, which allows the model to differentiate between 32 patterns. Next, we incorporate
two additional convolutional layers with 64 and 128 nodes
respectively. Following that, we flatten the layers and cre-
ate a dense layer similar to those in an Artificial Neural
Network (ANN). The purpose of this dense layer is to pro-
duce the final prediction.
9. Model compilation: During this step, you specify the
learning algorithm used and the techniques employed to
enhance the model’s performance. We have specified
the categorical cross-entropy loss function, which is com-
monly used for classifying non-binary models that differ-
entiate between three or more classes.
10. Model fitting: In this step, we initiate
the training process by inputting the data into the model
and performing forward propagation and backward propa-
gation multiple times. For this particular project, we con-
ducted the fitting process over eight epochs. To streamline
the training procedure and improve efficiency, we divided
the data into 391 batches, with each batch containing 128
samples. This batching approach reduces complexity and
enhances the overall runtime of the training process.
11. Model evaluation: During this step, we assess the perfor-
mance of the model by inputting the test data and making
predictions using the model. We then calculate the accu-
racy by comparing the predicted outputs with the actual
outputs in the test dataset. The accuracy is determined by
calculating the average error across the entire test dataset.
12. Save the model: In this step we save the model as a .h for-
mat so that each time we make predictions with the model
we do not have to train the model again.










