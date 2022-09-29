"""
선형 회귀 실습2
"""

import numpy as np

ab = [3,76]

# x, y의 데이터 값
data = [[2,81],[4,93],[6,91],[8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# y = ax + b에 a와 b 값을 대입하여 결과를 출력하는 함수
def predict(x):
    return ab[0]*x + ab[1]

# RMSE 함수
def rmse(p,a):
    return np.sqrt(((p - a) ** 2).mean())

#RMSE 함수를 각 y 값에 대입하여 최종 값을 구하는 함수
def rmse_val(predict_result, y):
    return rmse(np.array(predict_result), np.array(y))

#예측 값이 들어갈 빈 리스트
predict_result = []

# 모든 x 값을 한 번씩 대입하여
for i in range(len(x)):
    #predict_result 리스트를 완성 시키기
    predict_result.append(predict(x[i]))
    print("공부한 시간=%.f, 실제 점수=%.f, 예측 점수=%.f" %(x[i],y[i],predict(x[i])))
    predict(x[i])

#최종 RMSE 최종값 출력
print("rmse 최종값: " + str(rmse_val(predict_result,y)))