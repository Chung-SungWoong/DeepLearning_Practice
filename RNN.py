import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras import optimizers

def sin(x):
    return np.sin(2.0 * np.pi * x / 25)

def toy_problem(T):
    x = np.arange(0, 25*T + 1)
    return sin(x)

T = 100
f = toy_problem(T).astype(np.float32)
length_of_sequences = len(f)

maxlen = 25

x = []
t = []

for i in range(length_of_sequences - maxlen):
    x.append(f[i:i+maxlen])
    t.append(f[i+maxlen])
    
x = np.array(x).reshape(-1, maxlen, 1)
t = np.array(t).reshape(-1,1)

x_train, x_val, t_train, t_val = \
    train_test_split(x, t, test_size=0.2, shuffle= False)
    

model = Sequential()

model.add(SimpleRNN(50, activation = 'tanh'))
model.add(Dense(1,activation ='linear'))


optimizer = optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
hist = model.fit(x_train, t_train, epochs=100, batch_size=100, verbose=2, validation_data=(x_val, t_val))

prediction_T = 5
gen = [None for i in range(maxlen)]
z = x[:1]

for i in range(prediction_T*25+1):
    last_z = z[-1:]
    preds = model.predict(last_z)
    z1 = z
    z = np.append(z, preds)[1:]
    
    z = z.reshape(-1, maxlen,1)
    gen.append(preds[0,0])
    

sin = toy_problem(prediction_T+1)
fig = plt.figure()
plt.rc('font', family  ='serif')
plt.xlim([0,25*(prediction_T+1)])
plt.ylim([-1.5,1.5])
plt.plot(range(prediction_T*25+25+1), sin, color ='blue', linestyle = '--',linewidth=0.5)
plt.plot(range(prediction_T*25+25+1), gen, color='red', linewidth=1, marker= 'o', markersize =1, markerfacecolor='red', markeredgecolor='red')
plt.savefig('output.png')
plt.show()