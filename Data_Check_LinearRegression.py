"""
지금까지 실습의 예들의 데이터는 참 또는 거짓을 맞히는 문제, 아니면 여러 개의 보기 중 맞는 하나를 예측하는 문제
이번에는 수치를 예측하는 문제이다
= 수치를 예측하는 선형 회귀 문제
"""
import pandas as pd
df = pd.read_csv("/Users/chung_sungwoong/Desktop/Practice/DeepLearning_Practice/dataset/housing.csv", delim_whitespace=True, header=None)

print(df.info())
#print(df.head())