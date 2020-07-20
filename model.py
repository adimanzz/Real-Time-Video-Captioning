# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:03:59 2020

@author: aditya

"""


import keras
from keras import Input
from keras.layers.merge import add
from keras.models import Model 
from keras.layers import Bidirectional, LSTM, Dense, Dropout, Embedding
from attention import Attention


class CaptionNet():
    def __init__(self, max_length, vocab_size, embedding_dims):
        self.MAX_LENGTH = max_length
        self.VOCAB_SIZE = vocab_size
        self.embedding_dims = embedding_dims
        
    def forward(self):
        input1 = Input(shape = (2048,))
        fea = Dropout(0.4)(input1)
        fea = Dense(256, activation = 'relu')(fea)
        
        input2 = Input(shape = (self.MAX_LENGTH,))
        seq = Embedding(self.VOCAB_SIZE, self.embedding_dims, mask_zero = True)(input2)
        seq = Dropout(0.4)(seq)
        seq = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout = 0.1))(seq)
        seq = LSTM(256, return_sequences = True, recurrent_dropout = 0.1)(seq)
        seq = Attention()(seq)
        
        decoder = add([fea, seq])
        decoder = Dense(256, activation = 'relu')(decoder)
        outputs = Dense(self.VOCAB_SIZE, activation = 'softmax')(decoder)
        
        model = Model(inputs = [input1, input2], outputs = outputs)
         
        return model
    

     
                    
        