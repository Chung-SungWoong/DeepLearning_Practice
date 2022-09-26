"""
선형 회귀 실습2
"""

import numpy as np

ab = [3,76]

data = [[2,81],[4,93],[6,91],[8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

def predict(x):
    return ab[0]*x + ab[1]

def rmse(p,a):
    return np.sqrt(((p - a) ** 2).mean())

def