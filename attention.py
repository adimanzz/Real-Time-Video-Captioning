# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 09:52:12 2020

@author: aditya
"""
from keras.layers import Layer
import keras.backend as K

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1],1), initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1],1), initializer='zeros')
        super(Attention, self).build(input_shape)
        
    def compute_mask(self, input, input_mask=None):
    # do not pass the mask to the next layers
        return None    
        
    def call(self, x):
        at = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis = -1)
        at = K.softmax(at)
        at = K.expand_dims(at, axis = -1)
        output = x*at
        return K.sum(output, axis = 1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super(Attention, self).get_config()