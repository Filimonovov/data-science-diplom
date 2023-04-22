# GRNN -- обобщенно-регрессионная нейронная сеть

import pickle
import pandas as pd
import openpyxl
import numpy as np # библиотека для работы с матрицами
from scipy.optimize import minimize # Библиотека с методом оптимизации
import random as rnd
import math
import copy as cp
import sys

dest = 1 # Номер целевого столбца
columns = [2, 3, 4, 5, 6, 7, 10, 11, 12, 13] # Индексы столбцов со входными данными
k = 10 # Число ближайших соседей
AllowedPercent = 0.11

print("Внимание! Обучение может занять длительное время!")

def load_data():
  WS1 = pd.read_excel("X_bp.xlsx", skiprows=1, dtype=float)
  data = np.array(WS1);
  WS2 = pd.read_excel("X_nup.xlsx", skiprows=1, dtype=float)
  data2 = np.array(WS2);
  result = np.zeros((data.shape[0],14));
  for i in range(0,data.shape[0]):
      result[i,:] = np.append(data[i,:], data2[i,1:4])
  return result

def load_data_wrapper():
  table = load_data() # инициализация наборов данных
  maxs = [table[:,i].max() for i in range(0,14)]
  mins = [table[:,i].min() for i in range(0,14)]
  for i in range(1,14):
      table[:,i] = 2*(table[:,i] - mins[i])/(maxs[i] - mins[i]) - 1
  training_inputs = [np.reshape(np.append(t[2:8], t[10:14]), (10, 1)) for t in table]
  training_results = [t[dest] for t in table]
  v = list(zip(training_inputs,training_results,list(range(0,len(training_inputs)))))
  rnd.shuffle(v)
  training_inputs, training_results, training_idxs = zip(*v)
  with open("trainedSELGRNN.dat", 'wb') as f:
       pickle.dump(training_idxs,f) 
  n = len(training_inputs)
  validating_inputs = training_inputs[int(2*n/3):n]
  validating_results = training_results[int(2*n/3):n]
  training_inputs = training_inputs[0:int(2*n/3)]
  training_results = training_results[0:int(2*n/3)]
  return training_inputs, training_results, validating_inputs, validating_results, mins, maxs

# Эвклидово расстояние
def get_dist(x1, x2):
    d = np.square(x1 - x2)
    return np.sum(d)

xd, yd, xv, yv, mins, maxs = load_data_wrapper()

# Считаем матрицу квадратов расстояний
dists2 = [[0.0 for j in range(0, len(xd))] for i in range(0, len(xd))]
for i in range(0, len(xd)):
    for j in range(i+1, len(xd)):
        dists2[i][j] = get_dist(xd[i], xd[j])
        dists2[j][i] = dists2[i][j]

def goal(R): # Вводим целевую функцию для поиска коэффициента R. Цель -- получить погрешность AllowedPercent
    V = np.array(dists2)*R
    weights = np.exp(-V)
    predicted = np.dot(weights,yd)/np.dot(weights,np.ones((len(xd),1))).reshape((1,len(xd)))
    d = yd - predicted
    eps = np.mean(np.abs(d))
    return (eps - AllowedPercent)*(eps - AllowedPercent)

R = minimize(goal, 1.0).x
np.savetxt("TrainedGRNN_R.txt", R)

print("Найдено R = ", R)

V = np.array(dists2)*R
weights = np.exp(-V)
predicted = np.dot(weights,yd)/np.dot(weights,np.ones((len(xd),1))).reshape((1,len(xd)))
d = yd - predicted
print("Средняя ошибка = ", np.mean(np.abs(d)))

# Считаем матрицу квадратов расстояний
dists2 = [[0.0 for j in range(0, len(xd))] for i in range(0, len(xv))]
for i in range(0, len(xv)):
    for j in range(0, len(xd)):
        dists2[i][j] = get_dist(xv[i], xd[j])

V = np.array(dists2)*R
weights = np.exp(-V)
predicted = np.dot(weights,yd)/np.dot(weights,np.ones((len(xd),1))).reshape((1,len(xv)))
d = yv - predicted
print("Средняя ошибка по тестовым данным = ", np.mean(np.abs(d)))

