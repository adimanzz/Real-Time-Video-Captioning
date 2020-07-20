# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:22:31 2020

@author: aditya
"""
import cv2
import numpy as np
import pandas as pd

import keras
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from attention import Attention
from model import CaptionNet


model_encoder = InceptionV3(weights = 'imagenet')
model_encoder = Model(model_encoder.input, model_encoder.layers[-2].output)
target_size = (299,299)
MAX_LENGTH = 40
wordtoix = pd.read_pickle('word_index_dictionaries/wordtoix_58.pkl')
ixtoword = pd.read_pickle('word_index_dictionaries/ixtoword_58.pkl')


model = keras.models.load_model('Model/model6_bi.h5',
                                   custom_objects={'Attention':Attention})

def preprocess_image(image, target_size):
    #img = image.img_to_array(image)
    img = cv2.resize(image, target_size)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis = 0)
    return img


def decode(features):
    seq_pred = 'startseq'
    for i in range(MAX_LENGTH):
        sequence = [wordtoix[word] for word in seq_pred.split() if word in wordtoix]
        sequence = pad_sequences([sequence], maxlen = MAX_LENGTH)
        pred = model.predict([features, sequence])
        pred = np.argmax(pred)
        word = ixtoword[pred]
        seq_pred += ' ' + word
        if word == 'endseq':
            break
    final = seq_pred.split()
    final = final[1: -1]
    final = ' '.join(final)
    return final    



font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

while True:
    
    _, frame = cap.read()
    
    
    img = preprocess_image(frame, target_size)
    
    features = model_encoder.predict(img)
    caption = decode(features)
    
    cv2.putText(frame, caption , (10,450), font, 0.5, (255,255,255), 1)
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()    


























