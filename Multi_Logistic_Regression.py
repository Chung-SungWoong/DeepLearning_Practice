"""
다중 로지스틱 회귀
"""

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)


x_data = np.array([[2,3],[4,3],[6,4],[8,6],[10,7],[12,8],[14,9]])       # x의 데이터가 2차원 x[0] = 공부 시간, x[1] = 과외 횟수
y_data = np.array([0,0,0,1,1,1,1]).reshape(7,1)                         # y의 데이터는 True or False의 두가지 

X = tf.placeholder(tf.float64,shape=[None,2])
Y = tf.placeholder(tf.float64,shape=[None,1])

a = tf.Variable(tf.random.normal([2,1],dtype=tf.float64))
b = tf.Variable(tf.random.normal([1],dtype=tf.float64))

y = tf.sigmoid(tf.matmul(X,a) + b)

loss = -tf.reduce_mean(Y * tf.log(y) + (1 - Y) * tf.log(1 - y))

learning_rate = 0.1

gradient_descent = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)

predicted = tf.cast(y > 0.5, dtype = tf.float64)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float64))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(3001):
        a_, b_, loss_, _ = sess.run([a,b,loss,gradient_descent], feed_dict={X: x_data, Y:y_data})
        if (i+1) % 300 == 0:
            print("step= %d, a1 = %.4f, a2 = %.4f, b = %.4f, loss = %.4f" %(i + 1, a_[0], a_[1], b_, loss_))

    new_x = np.array([7,6.]).reshape(1,2)       
    new_y = sess.run(y, feed_dict= {X: new_x})

    print("공부한 시간: %d, 과외 수업 횟수: %d" % (new_x[:,0], new_x[:,1]))
    print("합격 가능성: %6.2f %%" %(new_y* 100))

"""
실제 값 적용하기
회귀 스크립트를 실제로 사용해 예측 값 구하기
"""