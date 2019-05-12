# Binary-Classifier
A Convolutional Neural Network to classify a image in 2 classes with accuracy over 86 percent over training set and 83 percent accuracy over test set.
 ## Libraries Required
1. Keras
2. numpy
3. Pickle
## Convolutional Neural Network
The Convolutional Neural Network(CNN) is a feed forward network applied to image data.Here Inputs are image's 3d matrix.So,basically our objective is to find patterns and trends in similar type of images and classify them as in the same class.
## Steps in Convolutional Neural Network
### Convolutional
This step involves moving a 3*3 martrix of feature detector across image matrix(multiply element wise) and then adding all 9 elements into one single entity of a new matrix. 
### Pooling
We are considering a pool of 2*2 matrix along the output of previous step and either taking average of 4 entities or taking mean of 4 entites into new matrix.
### Flattening
we convert our matrix into a single long column vector.
### Creating a full layered Artificial Neural Network 
Finally the column vector with important features extracted will act as a input for a fully connected layered Artificial Neural Network.
