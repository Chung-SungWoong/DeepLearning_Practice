
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

for train, test in skf.(X,Y):
    model = Sequential()
