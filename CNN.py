#Convolutional Neural Network

#Installing Theano
#Installing Tensorflow
#Installing Keras

#Part 1: Building the CNN
#importing Keras libraries and packages
#for intializing our Neural Network(as CNN is also sequence of layers)
from keras.models import Sequential
#To add Convolutional layer from input images
from keras.layers import Convolution2D
#To add Pooling layer from Convolutional layers
from keras.layers import MaxPooling2D
#To convert Pooled feature matrix into large vector(which is input for fully connected layers)
from keras.layers import Flatten
#To add Fully Connected layers in Our classic ANN
from keras.layers import Dense

#Intializing our CNN
classifier=Sequential()
#Step 1: Convolution(to extract features from different regions)
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#Step 2: Pooling(reducing feature map)
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Step 1: Convolution(Now adding Second Convolution layer)
classifier.add(Convolution2D(32,3,3,activation='relu'))
#Step 2: Pooling(For Scond Convolution layer)
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Step 3: Flattening(Converting pooled feature map into one long column)
classifier.add(Flatten())
#Step 4: Full Connection(Making a classic ANN with input as Flattened vector for image Prediction)
#adding hidden layer with 128 nodes
classifier.add(Dense(output_dim=128,activation='relu'))
#adding output layer with only 1 node
classifier.add(Dense(output_dim=1,activation='sigmoid'))

#Compiling the CNN
#optimizer is the alogrithm for finding global minima(adam from stochastic gradient descent)
#loss is the loss function which is cross entropy function 
#metrics is the performance metric
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Part 2 : Fitting the CNN to the images
#Importing ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
#Image Augmentation
train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

#Preprocess the images in test set
test_datagen=ImageDataGenerator(rescale=1./255)
#Generating Training Data set and Applying Image Augmentation
training_set=train_datagen.flow_from_directory('dataset/training_set',
                                               target_size=(64,64),
                                               batch_size=32,
                                               class_mode='binary')

#Generating Test Data set and Applying Image Augmentation
test_set=test_datagen.flow_from_directory('dataset/test_set',
                                          target_size=(64,64),
                                          batch_size=32,
                                          class_mode='binary')

#Fitting Training set and Evaluating Performance
classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)

#Part 3: Single Prediction
#Import numpy
import numpy as np
#Dealing with Single Image
from keras.preprocessing import image
test_image_1=image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
test_image_1=image.img_to_array(test_image_1)
test_image_1=np.expand_dims(test_image_1,axis=0)
result=classifier.predict(test_image_1)
training_set.class_indices
if result[0][0]>=0.5:
    prediction='dog'
else:
    prediction='cat'
print(prediction)
    
test_image_2=image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64,64))
test_image_2=image.img_to_array(test_image_2)
test_image_2=np.expand_dims(test_image_2,axis=0)
result=classifier.predict(test_image_2)
training_set.class_indices
if result[0][0]>=0.5:
    prediction='dog'
else:
    prediction='cat'
print(prediction)