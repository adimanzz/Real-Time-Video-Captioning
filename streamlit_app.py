# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:56:28 2020

@author: aditya
"""

import streamlit as st
import numpy as np
import pandas as pd
import time

import cv2
import numpy as np
from captions import *

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

import keras
from keras.models import Model
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from attention import Attention
#from model import CaptionNet

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

model_encoder = InceptionV3(weights = 'imagenet')
model_encoder = Model(model_encoder.input, model_encoder.layers[-2].output)
target_size = (299,299)

model2 = keras.models.load_model('Model/model2_bi.h5',
                                custom_objects={'Attention':Attention})

img_path = st.file_uploader('Drop an Image', type='jpg') 
MAX_LENGTH = 40
wordtoix = pd.read_pickle('wordtoix_28.pkl')
ixtoword = pd.read_pickle('ixtoword_28.pkl')

def preprocess_img(image_path, target_size):
    img = image.load_img(image_path, target_size)
    img = image.img_to_array(img)
    img = cv2.resize(img, target_size)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis = 0)
    return img

@st.cache(persist = True)
def decode(features):
    seq_pred = 'startseq'
    for i in range(MAX_LENGTH):
        sequence = [wordtoix[word] for word in seq_pred.split() if word in wordtoix]
        sequence = pad_sequences([sequence], maxlen = MAX_LENGTH)
        pred = model2.predict([features, sequence])
        pred = np.argmax(pred)
        word = ixtoword[pred]
        seq_pred += ' ' + word
        if word == 'endseq':
            break
    final = seq_pred.split()
    final = final[1: -1]
    final = ' '.join(final)
    return final  

def get_frame():
    return np.random.randint(0, 255, size=(10,10))

@st.cache(persist=True)
def generate_caption(img_path):
    img = preprocess_img(img_path, target_size)
    features = model_encoder.predict(img)            
    pred_caption = decode(features)        
    return pred_caption  
    

my_image = st.image(img_path, caption='Image', width=600)

if st.button('Generate Caption'):
    with st.spinner('Extracting Features...'):
        caption = generate_caption(img_path)  
      
    st.success('Done!')
    my_image.image(img_path, caption='Image', width=600)  
    st.title(caption)