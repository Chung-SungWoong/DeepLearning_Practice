"""
LSTM과 CNN을 사용하여 영화 리뷰 분류 데이터 뽑기
"""

import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM,Dropout, Activation
from keras.layers import Conv1D,MaxPooling1D


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 5000)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen= 100)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen= 100)

model = Sequential()
model.add(Embedding(5000,100))
model.add(Dropout(0.5))
model.add(Conv1D(64, 5, padding='valid', activation='relu', strides = 1))

model.add(MaxPooling1D(pool_size = 4))
model.add(LSTM(55))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

""" 출력값
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, None, 100)         500000    
                                                                 
 dropout (Dropout)           (None, None, 100)         0         
                                                                 
 conv1d (Conv1D)             (None, None, 64)          32064     
                                                                 
 max_pooling1d (MaxPooling1D  (None, None, 64)         0         
 )                                                               
                                                                 
 lstm (LSTM)                 (None, 55)                26400     
                                                                 
 dense (Dense)               (None, 1)                 56        
                                                                 
 activation (Activation)     (None, 1)                 0         
                                                                 
=================================================================
Total params: 558,520
Trainable params: 558,520
Non-trainable params: 0
_________________________________________________________________

MNIST 의 2차원 배열과 다르게 이 데이터들은 1차원 데이터를 다루고 있기때문에
Conv2D, MaxPooling2D 가 아닌 Conv1D MaxPooling1D 을 사용
"""
