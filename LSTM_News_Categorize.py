"""
LSTM을 이용하여 로이터 뉴스 카테고리 분류하기
"""
import numpy
from keras.utils import np_utils
import tensorflow as tf
from keras.datasets import reuters
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)
"""
데이터 체크
category = numpy.max(y_train) + 1
print(category, '카테고리')
print(len(x_train), '학습용 뉴스 기사')
print(len(x_test), '테스트용 뉴스 기사')
print(x_train[0])                       # 단어를 그대로 사용하지 않고 숫자로 변환한 다음 학습, 해당 단어가 몇 번이나 나타나는지 세어 빈도에 따라 번호를 붙임
"""

#x 데이터 단어 수를 100개로 맞추기 
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen= 100)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen= 100)

print(x_train)
print(x_test)
#y 데이터 원 핫 인코딩 처리하기
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Embedding(1000,100))
model.add(LSTM(100,activation='tanh'))
model.add(Dense(46,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(x_train,y_train, batch_size = 100, epochs = 20, validation_data=(x_test, y_test))

print("\n Test Accuracy: %.f" % (model.evaluate(x_test,y_test)[1]))
