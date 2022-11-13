"""
선형 회귀 데이터는 마지막에 참과 거짓을 구분할 필요가 없다 
즉, 출력층에 활성화 함수를 지정할 필요도 없다

모델의 학습을 확인하기 위해 예측 값과 실제 값을 비교하는 부분을 추가할 필요 있다

# flatten()은 데이터 배열이 몇 차원이든 모두 1차원으로 바꿔 읽기 쉽게 해 주는 함수
"""

from keras.models import Sequential
from keras.layers import Dense 
from sklearn.model_selection import train_test_split 

import numpy
import pandas as pd
import tensorflow as tf

seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv('/Users/chung_sungwoong/Desktop/Practice/DeepLearning_Practice/dataset/housing.csv', delim_whitespace=True, header=None)

dataset = df.values
x = dataset[:,0:13]
y = dataset[:,13]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(30,input_dim = 13, activation = 'relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(1))

model.compile(loss = 'mean_squared_error', optimizer = 'adam')

model.fit(x_train, y_train, epochs = 200, batch_size = 10)

y_prediction = model.predict(x_test).flatten()

for i in range(10):
    label = y_test[i]
    prediction = y_prediction[i]
    print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label,prediction))