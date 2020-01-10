# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:06:37 2020

@author: sehad
"""


import cv2
from keras.models import load_model
import tensorflow as tf



image = cv2.imread("D:Desktop/cnn/cap.png",0)
cv2.imshow('image',image)

scale_percent = 70 # percent of original size
width = int(image.shape[1] * scale_percent / 1070)
height = int(image.shape[0] * scale_percent / 780) 
dim = (width, height)
# resize image
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)

resized.reshape(28, 28)
image = image.astype('float32')
image /= 255


cv2.imshow("Resized image", resized)

model = load_model('my_model.h5')
pred = model.predict(resized.reshape(1, 28, 28, 1))
print(pred.argmax())

converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file("my_model.h5")

tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
