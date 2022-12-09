"""
컨볼루션 층을 통해 이미지 특징을 도출
하지만 아직 결과가 너무 크고 복잡아여 이를 다시 한번 더 축소
이 과정을 pooling 혹은 sub sampling이라고 함
풀링 기법 중 가장 많이 사용되는 방법이 맥스 풀링
맥스 풀링 - 정해진 구역 안에서 가장 큰 값만 다음 층으로 넘기고 나머지는 버리는 기법
"""

#CNN
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1),activation = 'relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))

#MaxPooling
model.add(MaxPooling2D(pool_size=2))

"""
과적합을 피하는 방법으로 간단하지만 효과가 큰 기법이 바로 드롭아웃(drop out)
드롭아웃 - 은닉층에 배치된 노드 중 일부를 임의로 꺼주는 것 
"""

#Drop out
model.add(Dropout(0.25))    # 25%의 노드를 끄기

#2차원 배열을 다시 1차원으로 바꿔주는 함수가 Flatten()
model.add(Flatten())