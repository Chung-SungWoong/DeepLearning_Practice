"""
코딩으로 확인하는 로지스틱 회귀
"""

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

data = [[2,0],[4,0],[6,0],[8,1],[10,1],[12,1],[14,1]]   # x와 y값을 정해준다
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

a = tf.Variable(tf.random.normal([1],dtype=tf.float64,seed=0))  # a와 b의 값을 임의로 설정
b = tf.Variable(tf.random.normal([1],dtype=tf.float64,seed=0))

y = 1/(1 + np.e**(a * x_data + b))                      # 시그모이드 함수 방정식

loss = -tf.reduce_mean(np.array(y_data) * tf.math.log(y) + (1 - np.array(y_data)) * tf.math.log(1 - y))     # loss functionㅇ

learning_rate = 0.5                                     # 학습률 설정
gradient_descent = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)    # 경사 하강법을 통한 오차 최소화하기 

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())   #초기화 후 

    for i in range(60001):                              
        sess.run(gradient_descent)
        if i % 6000 == 0:
            print("Epoch: %.f, loss = %.4f, 기울기 a = %.4f, y 절편 = %.4f" %(i,sess.run(loss),sess.run(a),sess.run(b)))