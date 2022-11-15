"""
0 - 9 까지 손글씨 데이터 셋
"""
from keras.datasets import mnist
import matplotlib.pyplot as plt
import sys

(x_train, y_class_train),(x_test, y_class_test) = mnist.load_data()

#plt.imshow(x_train[0], cmap='Greys')              글자 하나 체크
#plt.show()

#print("학습셋 이미지 수: %d 개" %(x_train.shape[0]))
#print("테스트셋 이미지 수: %d 개" %(x_test.shape[0]))


"""
for x in x_train[0]:                                # 그림의 데이터 형식 확인 
    for i in x:
        sys.stdout.write('%d\t' %i)
    sys.stdout.write('\n')
"""

x_train = x_train.reshape(x_train.shape[0], 784)