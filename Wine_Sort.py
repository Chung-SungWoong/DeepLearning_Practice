from keras.models import Sequential 
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy
import tensorflow as tf
import os

seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

df_pre = pd.read_csv('/Users/chung_sungwoong/Desktop/Practice/DeepLearning_Practice/dataset/wine.csv', header=None)
df = df_pre.sample(frac=1)

dataset = df.values

X = dataset[:,0:12]
Y = dataset[:,12]

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

# 모델 업데이트 하기.
# 단순히 모델을 저장하는 것이 아니라 에포크마다 모델의 정확도 함께 기록하면서 저장

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'       # 모델을 epoch00-val_loss0000.hdf5 형식으로 저장

checkpointer = ModelCheckpoint(filepath=modelpath, monitor = 'val_loss',verbose=1)  # 모니터할 값을 지정, acc = 학습 정확도, val_acc = 테스트셋 정확도, loss = 학습셋 오차
# 모델이 저장될 곳을 modelpath로 지정하고 verbose 값을 1로 하면 해당 함수의 진행 사항이 출력된다

# checkpointer = ModelCheckpoint(filepath = modelpath, monitor= 'val_loss',verbose=1, save_best_only = True)   모델이 앞서 저장한 모델보다 나아졌을때만 저장하게 할때는 save_best_only를 사용한다  

model.fit(X,Y,validation_split = 0.2, epochs = 200, batch_size = 200, verbose = 0, callbacks=[checkpointer])

print("\n Accuracy: %.4f" %(model.evaluate(X,Y)[1]))



