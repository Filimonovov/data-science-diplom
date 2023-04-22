# GRNN -- обобщенно-регрессионная нейронная сеть

import pickle
import pandas as pd
import openpyxl
import numpy as np # библиотека для работы с матрицами
import math
import copy as cp
import sys

columns = [2, 3, 4, 5, 6, 7, 10, 11, 12, 13] # Индексы столбцов со входными данными
k = 3 # Число ближайших соседей

if len(sys.argv) < 11:
   print("Использование: 'predict_3.py' далее через пробел в командной строке указываются:")
   print("<Плотность, кг/м3>")
   print("<модуль упругости, ГПа>")
   print("<Количество отвердителя, м.%>")
   print("<Содержание эпоксидных групп,%_2>")
   print("<Температура вспышки, С_2>")
   print("<Поверхностная плотность, г/м2>")
   print("<Потребление смолы, г/м2>")
   print("<Угол нашивки, град>")
   print("<Шаг нашивки>")
   print("<Плотность нашивки>")
   print("Пример: predict_3.py 1880 313 129 21.25 300 210 220 0 10 57")
   exit()

# Загружаем подобранный коэффициент
R = np.loadtxt("TrainedGRNN_R.txt")

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
  with open("trainedSELGRNN.dat", 'rb') as f:
       training_idxs = pickle.load(f)
  training_inputs = [training_inputs[i] for i in training_idxs]
  training_results = [training_results[i] for i in training_idxs]
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

dest = 1 # Номер целевого столбца
xd, yd, xv, yv, mins, maxs = load_data_wrapper()

x = np.zeros((1, 10))
for i in range(0,10):
    x[0,i] = 2.0*(float(sys.argv[i+1]) - mins[columns[i]])/(maxs[columns[i]] - mins[columns[i]]) - 1.0

# Считаем отклик сети
dists2 = [get_dist(x.reshape((10,1)), xd[i]) for i in range(0, len(xd))] # Находим расстояния до всех образцовых точек
weights = np.exp(-R*np.array(dists2))
predicted = np.sum(yd*weights)/np.sum(weights)
predicted = 0.5*(predicted + 1.0)*(maxs[dest] - mins[dest]) + mins[dest]
print("Соотношение матрица-наполнитель = ", predicted)
