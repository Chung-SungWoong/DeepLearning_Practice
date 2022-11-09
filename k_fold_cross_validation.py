
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import pandas as pd 
import numpy
import tensorflow as tf


seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv('/Users/chung_sungwoong/Desktop/Practice/DeepLearning_Practice/dataset/sonar.csv', header=None)


dataset = df.values
x = dataset[:,0:60].astype(float)       
y_obj = dataset[:,60]

e = LabelEncoder()
e.fit(y_obj)
y = e.transform(y_obj)


n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle= True, random_state = seed)

accuracy = []

for train, test in skf.split(x,y):
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='mean_squared_error',optimizer = 'adam', metrics= ['accuracy'])
    model.fit(x[train], y[train], epochs = 100, batch_size = 5)
    k_accuracy = "%.4f" %(model.evaluate(x[test],y[test])[1])
    accuracy.append(k_accuracy)

print("\n %.f fold accuracy: " %n_fold, accuracy)