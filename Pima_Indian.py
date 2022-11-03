"""
Expect Pima Indian 
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf

seed = 0 
numpy.random.seed(seed)
tf.set_random_seed(seed)

dataset = numpy.loadtxt("",delimiter = ',')
x = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()
model.add(Dense(12,input_dim=8, activation='relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x,y,epochs = 200, batch_size = 10)

print("\n Accuracy: %.4f" %(model.evaluate(x,y)[1]))
