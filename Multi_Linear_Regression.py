"""
다중 선형 회귀 실습
x2 = 과외 시간 횟수를 넣음으로써 
기울기 a 를 한개 더 추가하여 1차원 예측 직선에서 3차원 예측 평면으로 바꿈
"""

import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

data = [[2,0,81],[4,4,93],[6,2,91],[8,3,97]]
x1 = [x_row1[0] for x_row1 in data]
x2 = [x_row2[1] for x_row2 in data]
y_data = [y_row[2] for y_row in data]

a1 = tf.Variable(tf.random.uniform([1],0,10,dtype=tf.float64,seed=0))
a2 = tf.Variable(tf.random.uniform([1],0,10,dtype=tf.float64,seed=0))
b = tf.Variable(tf.random.uniform([1],0,100,dtype=tf.float64,seed=0))

y = a1 * x1 + a2 * x2 + b       # 새로운 방정식 세우기 

rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

learning_rate = 0.1

gradient_descent = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):
        sess.run(gradient_descent)
        if step % 100 == 0:
            print("Epoch: %.f, RMSE = %.04f, 기울기 a1 = %.4f, 기울기 a2 = %.4f, y 절편 b = %.4f" %(step, sess.run(rmse), sess.run(a1), sess.run(a2),sess.run(b)))