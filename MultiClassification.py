"""
다중 분류 문제
클래스가 3개 이상일때 사용 가능.
참 거짓을 해결하는 것이 아니라 여러개중 어느것이 답인지 예측하는 문제

예제에서는 아이리스의 품종을 예측하는 모델
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#df = pd.read_csv('/Users/chung_sungwoong/Desktop/Practice/DeepLearning_Practice/dataset/iris.csv', names = ['special_length','sepal_width','petal_length','petal_width','species'])
df = pd.read_excel('/Users/chung_sungwoong/Desktop/Check2.xlsx', names = ['accuracy','time','user'])
#print(df.head())
sns.pairplot(df,hue='user')
plt.show()
