"""
초음파 광물 예측하기
"""

from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd 
import numpy
import tensorflow as tf

seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv('/Users/chung_sungwoong/Desktop/Practice/DeepLearning_Practice/dataset/sonar.csv', header=None)

dataset = df.values
x = dataset[:,0:60].astype(float)       # tensorflow 2로 넘어가면서 float으로 형변환을 제대로 해줘야 함
y_obj = dataset[:,60]

e = LabelEncoder()
e.fit(y_obj)
y = e.transform(y_obj)

model = Sequential()
model.add(Dense(24, input_dim = 60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer = 'adam', metrics= ['accuracy'])

model.fit(x,y,epochs = 200, batch_size = 5)

print("\n Accuracy: %.4f" %(model.evaluate(x,y)[1]))
