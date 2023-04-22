import pickle
import pandas as pd
import openpyxl
import numpy as np # библиотека для работы с матрицами
import math
import copy as cp
import sys

n_clusters = 25 # Число кластеров
nump = 9 # Количество коэффициентов для каждого частного геометрического полинома
EPS = 0.0001 # Допустимая погрешность
k = 5 # Параметр метода k-ближайшего соседа
columns = [2, 3, 4, 5, 6, 7, 10, 11, 12, 13] # Индексы столбцов со входными данными

if len(sys.argv) < 11:
   print("Использование: 'predict_2.py' далее через пробел в командной строке указываются:")
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
   print("Пример: predict_2.py 1880 313 129 21.25 300 210 220 0 10 57")
   exit()

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
  with open("trainedSEL." + str(dest) + ".dat", 'rb') as f:
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

def trigo_interp(a, xx): # Интерполирующая частичная тригонометрическая функция
    arg1 = (a[1]*xx + a[2])
    arg2 = (a[4]*xx + a[5])
    arg3 = (a[7]*xx + a[8])
    return a[0]*np.cos(arg1)+a[3]*np.cos(arg2)+a[6]*np.cos(arg3)

def full_interp(c, ir, k, b, x): # Полная интерполирующая функция на кластере
    r = np.zeros((x.shape[0], 1))
    for i in range(0, ir):
        xx = x[:,i]
        r += 0.5*(trigo_interp(c[i],xx) + 1.0)/k[i] + b[i]
    return r

for dest in range(8,10): # Номер целевого столбца
    xd, yd, xv, yv, mins, maxs = load_data_wrapper()
    # Загружаем данные обученного алгоритма
    with open("trainedIDXS." + str(dest) + ".dat", 'rb') as f:
        idxs = pickle.load(f) 
    with open("trainedK." + str(dest) + ".dat", 'rb') as f:
        kt = pickle.load(f) 
    with open("trainedB." + str(dest) + ".dat", 'rb') as f:
        bt = pickle.load(f) 
    with open("trainedC." + str(dest) + ".dat", 'rb') as f:
        ct = pickle.load(f) 
    with open("trainedIR." + str(dest) + ".dat", 'rb') as f:
        ir = pickle.load(f)
    x = np.zeros((1, 10))
    for i in range(0,10):
        x[0,i] = 2.0*(float(sys.argv[i+1]) - mins[columns[i]])/(maxs[columns[i]] - mins[columns[i]]) - 1.0
    # Методом k-ближайшего соседа находим нужный кластер
    dists = [get_dist(x.reshape((10,1)), xd[i]) for i in range(0, len(xd))] # Находим расстояния до всех образцовых точек
    votes = [0 for i in range(0, n_clusters)] # Голоса, поданные за кластеры
    for m in range(0,k):
        min_i = 0
        for i in range(1, len(xd)):
            if dists[i] < dists[min_i]:
               min_i = i
        votes[idxs[min_i]] += 1
        dists[min_i] = 10000000 # Чтобы больше не считать этого найденного соседа ближайшим
    max_vote = max(votes)
    # Ищем кластер с наибольшим числом голосов "За"
    for c in range(0, n_clusters):
        if votes[c] == max_vote:
           clust = c
           break
    print("Номер кластера = ", clust)
    base = clust*(xd[0].size+1)
    predicted = full_interp(ct[clust],ir[clust],kt[base:base+xd[0].size+1],bt[base:base+xd[0].size+1],x)
    predicted = 0.5*(predicted + 1.0)*(maxs[dest] - mins[dest]) + mins[dest]
    if dest == 8:
       print("Модуль упругости при растяжении, ГПа = ", predicted)
    elif dest == 9:
       print("Прочность при растяжении, МПа = ", predicted)
