from pyexpat import model
from keras.models import Sequential
from keras.layers import Dense

# Sequential 함수는 딥러닝의 구조를 한 층 한 층 쉽게 쌓아 올릴 수 있게 해준다.
# Sequential 함숳를 선언 후 model.add() 함수를 사용하여 필요한 층을 차례로 추가해준다

model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# activation = 다음 층으로 어떻게 값을 넘길지 결정하는 부분.

# 층을 몇 개 쌓을지는 데이터에 따라 그때 결정한다.
# model.add() 함수 안에 있는 Dense() 함수는 각 층이 어떤 특성을 가질 지 옵션을 설정하는 역할을 한다.

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# loss = 한 번 신경망이 실행될 때마다 오차 값을 추적하는 함수
# optimizer = 오차를 어떻게 줄여 나갈지 정하는 함수
model.fit(X,Y, epochs=30, batch_size = 10)

print("\n Accuracy: %.4f" %(model.evaluate(X,Y)[1]))
# 출력부분에서 model.evaluate() 함수를 이용해 앞서 만든 딥러닝의 모델이 어느 정도 정확하게 예측하는지 점검 가능
