
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf

(x_train, y_train),(x_test, y_test) = mnist.load_data()     #x,y의 트레이닝 셋과 테스트셋 가져오기

x_train = x_train.reshape(x_train.shape[0],784).astype('float32') /255  # 최적의 성능을 위해 수들을 255로 나눠주기
x_test = x_test.reshape(x_test.shape[0],784).astype('float32') / 255    # x 테스트 셋도 똑같은 방식으로 숫자 나눠주기

y_train = np_utils.to_categorical(y_train,10)                # y 값을 원 핫 인코딩으로 리스트 값으로 바꿔주기 
y_test = np_utils.to_categorical(y_test,10)              

#모델 최적화
model = Sequential()
model.add(Dense(512,input_dim=784, activation= 'relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# 모델 디렉토리 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# 모델 저장
modelpath='./mode/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath=modelpath,monitor='val_loss',verbose=1, save_best_only = True)      # verbose의 값이 1이면 함수 진행이 출력되고, 0이면 출력되지 않는다
early_stopping_callback = EarlyStopping(monitor = 'val_loss',patience = 10)


# 모델 실행
history = model.fit(x_train,y_train,validation_data = (x_test, y_test), epochs= 30, batch_size = 200, verbose =0, callbacks=[early_stopping_callback, checkpointer])

print("\n Test Accuracy: %.4f" %(model.evaluate(x_test,y_test)[1]))         # 테스트 셋으로 정확도 출력하기

#테스트셋 오차
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

# 비쥬얼로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len,y_vloss, marker = ',',c='red',label = 'Testset_loss')
plt.plot(x_len,y_loss, marker = '.', c='blue', label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()