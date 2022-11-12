"""
데이터  확인과 실행
"""


import pandas as pd

df_pre = pd.read_csv('/Users/chung_sungwoong/Desktop/Practice/DeepLearning_Practice/dataset/wine.csv', header = None)
df = df_pre.sample(frac=1)

print(df.head(5))

print(df.info())        # 총 6497개의 샘플이 있고 13개의 속성이 무엇인지 알 수 있음