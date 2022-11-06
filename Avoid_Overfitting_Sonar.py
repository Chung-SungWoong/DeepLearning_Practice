"""
초음파 광물 예측하기
Overfitting
모델이 학습 데이터셋 안에서는 일정 수준 이상의 예측 정확도를 보이지만 새로운 데이터에 적용하면 맞는 않는 것
층이 너무 많거나 변수가 복잡해서 발생하거나 테스트셋과 학습 셋이 중복될 때 생기기도 한다
학습셋 내부에서 성공률이 높아져도 테스트셋에서 효과가 없다면 과적합일 가능성이 높다
"""

from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)     #학습셋과 테스트셋의 구분

model = Sequential()
model.add(Dense(24, input_dim = 60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer = 'adam', metrics= ['accuracy'])

model.fit(x,y,epochs = 130, batch_size = 5)

print("\n Accuracy: %.4f" %(model.evaluate(x,y)[1]))
