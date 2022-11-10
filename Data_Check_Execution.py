"""
데이터  확인과 실행
"""

import pandas as pd

df_pre = pd.read_csv('/Users/chung_sungwoong/Desktop/Practice/DeepLearning_Practice/dataset/wine.csv', header = None)
df = df_pre.sample(frac=1)

print(df.head(5))