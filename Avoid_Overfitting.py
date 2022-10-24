"""
과적합 피하기
과적합은 층이 너무 많거나 변수가 복잡해서 발생하거나 테스트셋과 학습셋이 중복될 때 생기기도 한다.
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

df = pd.read_csv('/Users/chung_sungwoong/Desktop/Practice/DeepLearning_Practice/dataset/sonar.csv', header = None)

dataset = df.values
X = dataset[:,0:60].astype(float)           # int 에서 float으로 변환 필
Y_OBJ = dataset[:,60]

e = LabelEncoder()
e.fit(Y_OBJ)
Y = e.transform(Y_OBJ)

model = Sequential()
model.add(Dense(24,input_dim=60, activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


model.compile(loss='mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X,Y,epochs = 200, batch_size=5)

print("\n Accurcay: %.4f" %(model.evaluate(X,Y)[1]))