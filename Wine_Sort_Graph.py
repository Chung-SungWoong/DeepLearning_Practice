"""
그래프로 확인
모델의 학습 시간에 따른 정확도와 테스트 결과를 그래프를 통해 확인
"""
from keras.models import Sequential 
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import pandas as pd
import numpy
import tensorflow as tf
import os


seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

df_pre = pd.read_csv('/Users/chung_sungwoong/Desktop/Practice/DeepLearning_Practice/dataset/wine.csv', header=None)
df = df_pre.sample(frac= 0.15)

dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])


MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5' 
checkpointer = ModelCheckpoint(filepath=modelpath, monitor = 'val_loss',verbose=1) 

history = model.fit(X,Y,validation_split = 0.33, epochs = 1500, batch_size = 500)   # 모델 실행 및 저장

y_vloss=history.history['val_loss']     # 실험 오차값 저장

y_acc = history.history['accuracy']          # 실험 정확도 값 저장

x_len = numpy.arange(len(y_acc))       # x 값을 저장

plt.plot(x_len,y_vloss, "o", c='red', markersize = 3)
plt.plot(x_len,y_acc, "o", c='blue', markersize = 3)

plt.show()

"""
학습이 진행될수록 정확도는 올라가지만 과적합으로 실험 결과는 점점 나빠지게 된다
이렇게 테스트셋 오차가 줄지 않으면 학습을 멈추게 하는 함수가 있다
from keras.callbacks immport EarlyStopping
EarlyStopping()
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)

model.fit(X,Y, validation_split = 0.33, epochs = 3500, batch_size = 500, callbacks = [early_stopping_callback])
"""
