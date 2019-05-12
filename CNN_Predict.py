#Part 3: Single Prediction
#Import numpy
import numpy as np
from pickle import load
from keras.models import load_model
#loading model
classifier=load_model('model.h5')
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
if result[0][0]>0.5:
    prediction='dog'
else:
    prediction='cat'
print(prediction)