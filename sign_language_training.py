import pandas as pd 
import numpy as np 
import os
import cv2
import matplotlib.pyplot as plt
import warnings
import re
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow import keras
import tensorflow as tf
import PIL
from sklearn.metrics import classification_report, confusion_matrix
from sklearn_evaluation import plot
import itertools

### seed belirleme
seed = 1842
tf.random.set_seed(seed)
np.random.seed(seed)
warnings.simplefilter('ignore')


# veri seti tanimlama
image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)


train_data_gen = image_generator.flow_from_directory(directory="Sign Language for Numbers",
                                                      target_size=(100, 100), 
                                                      color_mode='grayscale',
                                                      class_mode='categorical',
                                                      shuffle=True,
                                                      subset='training')

val_data_gen = image_generator.flow_from_directory(directory="Sign Language for Numbers",
                                                    target_size=(100, 100),
                                                    color_mode='grayscale',
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    subset='validation')

                                     
# model olusturma
def get_model():
  inputs = Input(shape=(100, 100, 1))

  conv_1 = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same')(inputs)
  act_1 = tf.keras.layers.Activation('relu')(conv_1)
  pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(act_1)

  conv_2 = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), padding='same')(pool_1)
  act_2 = tf.keras.layers.Activation('relu')(conv_2)
  pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(act_2)

  conv_3 = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1), padding='same')(pool_2)
  act_3 = tf.keras.layers.Activation('relu')(conv_3)
  pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(act_3)

  conv_4 = tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), padding='same')(pool_3)
  act_4 = tf.keras.layers.Activation('relu')(conv_4)
  pool_4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(act_4)

  flatten = tf.keras.layers.Flatten()(pool_4)

  dense_1 = tf.keras.layers.Dense(512)(flatten)
  act_5 = tf.keras.layers.Activation('relu')(dense_1)
  drop_1 = tf.keras.layers.Dropout(0.5)(act_5)

  dense_2 = tf.keras.layers.Dense(128)(drop_1)
  act_6 = tf.keras.layers.Activation('relu')(dense_2)
  drop_2 = tf.keras.layers.Dropout(0.5)(act_6)

  dense_3 = tf.keras.layers.Dense(10)(drop_2)
  act_7 = tf.keras.layers.Activation('softmax')(dense_3)

  model = tf.keras.Model(inputs=inputs, outputs=act_7)

  return model

model = get_model()

model.summary()

# modeli derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 'acc'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# modeli eğitme
history = model.fit_generator(train_data_gen, epochs=100, validation_data=val_data_gen, callbacks=[callback], workers=4)

# modeli kaydetme
model.save('isaret_dili_model.h5')

# modeli yükleme
model = tf.keras.models.load_model('isaret_dili_model.h5')

# modeli degerlendirme
def model_evaluation(model, test):
  test_loss, test_acc, *is_anything_else_being_returned = model.evaluate(test)
  print('Test accuracy:', test_acc)
  print('Test loss:', test_loss)
  return [test_loss, test_acc]

model_evaluation(model, val_data_gen)

# tahmin yapma
def predict(path):
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img, (100, 100))
  img = np.array(img).reshape(-1, 100, 100, 1)
  img = img/255
  pred = model.predict(img)
  return str(np.argmax(pred, axis=1)[0])

predictions = []

for i in range(10):
  path = r"C:\Users\PC\Desktop\el_deneme\emiron\{}.jpg".format(i)
  predictions.append(predict(path))

# tahminleri görselleştirme
def plot_predictions(predictions):
    fig = plt.figure(figsize=(9, 9))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(cv2.resize(cv2.imread(r"C:\Users\PC\Desktop\el_deneme\emiron\{}.jpg".format(i))[:,:,::-1], (150,150)))
        plt.title(predictions[i])
        plt.axis('off')
    plt.show()

plot_predictions(predictions)
