# Databricks notebook source
import cv2

# COMMAND ----------

photo = cv2.imread("vk.jpg")

# COMMAND ----------

photo.shape

# COMMAND ----------

feature = 480*640

# COMMAND ----------

feature

# COMMAND ----------

no_ofweight = 307200

# COMMAND ----------

photo

# COMMAND ----------

from keras.models import Sequential

# COMMAND ----------

model = Sequential()

# COMMAND ----------

model.get_config()

# COMMAND ----------

from keras.layers import Convolution2D

# COMMAND ----------

model.add(
Convolution2D(
    filters=32,
    kernel_size=(3,3),
    input_shape=(64,64,3),
    activation='relu'
)
)

# COMMAND ----------

model.get_config()

# COMMAND ----------

from keras.layers import MaxPooling2D

# COMMAND ----------

model.add(
MaxPooling2D(pool_size=(2,2))
)

# COMMAND ----------

model.get_config()

# COMMAND ----------

from keras.layers import Flatten

# COMMAND ----------

model.add(Flatten())

# COMMAND ----------

model.get_config()

# COMMAND ----------

from keras.layers import Dense

# COMMAND ----------

model.add ( Dense(units=128, activation='relu'))

# COMMAND ----------

model.get_config()

# COMMAND ----------

model.add ( Dense(units=1, activation='sigmoid'))

# COMMAND ----------

model.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['accuracy'])

# COMMAND ----------

from keras_preprocessing.image import ImageDataGenerator

# COMMAND ----------

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# COMMAND ----------

train_datagen.class_indices

# COMMAND ----------



# COMMAND ----------

 model.fit(
        train_generator,
        steps_per_epoch=8000,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=800)

# COMMAND ----------

# model.save('newmodel.h5')

# COMMAND ----------

from keras.models import load_model

# COMMAND ----------

model = load_model('cnn-cat-dog-model.h5')

# COMMAND ----------

from keras.preprocessing import image

# COMMAND ----------

test_img = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg')

# COMMAND ----------

test_img

# COMMAND ----------

type(test_img)

# COMMAND ----------

model.predict(test_img)

# COMMAND ----------

test_img_arr = image.img_to_array(test_img)

# COMMAND ----------

type(test_img_arr)

# COMMAND ----------

test_img_arr.shape

# COMMAND ----------

model.predict(test_img)

# COMMAND ----------

import numpy

# COMMAND ----------

test_image_arr_4d = numpy.expand_dims(test_img, axis=0)

# COMMAND ----------

test_image_arr_4d.shape

# COMMAND ----------

model.predict(test_image_arr_4d)

# COMMAND ----------

model.get_config()

# COMMAND ----------


