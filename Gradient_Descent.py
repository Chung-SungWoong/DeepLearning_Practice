import tensorflow as tf 

tf.compat.v1.disable_v2_behavior() # tensorflow 2.0 으로 업그레이드 됨에 따른 오류 방지용

# 케라스는 많은 개념을 자동으로 실행하게끔 되어있다
# 그래서 딥러닝의 동작 원리를 배울때는 tensorflow와 파이썬 만을 사용

data= [[2,81],[4,93],[6,91],[8,97]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

learning_rate = 0.1

a = tf.Variable(tf.random.uniform([1],0, 10, dtype = tf.float64, seed = 0))
b = tf.Variable(tf.random.uniform([1],0,100,dtype=tf.float64, seed = 0))

# random_uniform은 임의의 수를 생성해 주는 함수로 몇개의 값을 뽑아낼지와 최솟값 및 최댓갑을 적어준다
# tf.random_uniform([1],0,10...) 의 뜻은 0에서 10 사이의 수 중에서 임의의 수 1개를 만들라는 뜻

#일차 방정식 구현
y = a * x_data + b

# 평균 제곱근 오차의 식 구현
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

# 경사하강법
gradient_descent = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(rmse)



with tf.compat.v1.Session() as sess:
    #변수 초기화
    sess.run(tf.compat.v1.global_variables_initializer())
    #2001번 실행
    for step in range(2001):
        sess.run(gradient_descent)
        #100번마다 결과 출력
        if step % 100 == 0:
            print("Epoch: %.f, Rmse = %.04f, 기울기 a = %.4f, y 절편 b = %.4f" %(step, sess.run(rmse), sess.run(a), sess.run(b)))

# 결과값으로 RMSE의 변화와 기울기 a가 2.3에 수렴하는 것 그리고 y절편 b가 79에 수렴하는 과정을 볼 수 있다