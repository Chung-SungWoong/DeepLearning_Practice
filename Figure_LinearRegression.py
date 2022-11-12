"""
선형 회귀 데이터는 마지막에 참과 거짓을 구분할 필요가 없다 
즉, 출력층에 활성화 함수를 지정할 필요도 없다

모델의 학습을 확인하기 위해 예측 값과 실제 값을 비교하는 부분을 추가할 필요 있다

# flatten()은 데이터 배열이 몇 차원이든 모두 1차원으로 바꿔 읽기 쉽게 해 주는 함수
"""

from keras.models import Sequential
from keras.layers import Dense 
from sklearn.model_selection import train_test_split 

import numpy
import pandas as pd
import tensorflow as tf



