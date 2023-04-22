import pickle
import pandas as pd
import openpyxl
import numpy as np # библиотека для работы с матрицами
from scipy.optimize import minimize # Библиотека с методом оптимизации
import math
import random as rnd
import copy as cp
import sys

if len(sys.argv) < 2:
   print("Использование для обучения по модулю упругости: teach_2.py module")
   print("Использование для обучения по прочности: teach_2.py strength")
   exit()

if sys.argv[1] == "module":
   dest = 8; # Номер целевого столбца
elif sys.argv[1] == "strength":
   dest = 9; # Номер целевого столбца
else:
   print("Непонятный параметр. Должно быть 'module' или 'strength'")
   exit()

print("Внимание! Обучение может занять длительное время!")

n_clusters = 25 # Число кластеров
nump = 9 # Количество коэффициентов для каждого частного геометрического полинома
dv = 0.00001 # Шаг вычисления производных
EPS = 0.0001 # Допустимая погрешность
km = 5 # Параметр метода ближайшего соседа

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
  with open("trainedSEL." + str(dest) + ".dat", 'wb') as f:
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

# Кластеризация алгоритмом k-means
def clusterize_kmeans(x, nc):
  centers = [np.array(rnd.choice(x)) for i in range(0, nc)]
  idxs = [0 for i in x]
  delta = 1000
  while (delta > EPS): # Пока центры кластеров не перестали изменяться
    for i in range(0, len(x)): # Классифицируем каждую точку
        idxs[i] = 0 # Сначала считаем ближайшим нулевой кластер
        min_dist = get_dist(x[i], centers[0]) # Расстояние до него
        for j in range(1, nc): # Находим кластер с минимальным расстоянием до центра
            cur_dist = get_dist(x[i], centers[j])
            if cur_dist < min_dist:
               idxs[i] = j
               min_dist = cur_dist
    # Перевычисляем центры кластеров
    old = cp.copy(centers)
    centers = [np.zeros((x[0].size,1)) for i in range(0, nc)]
    sizes = [0 for i in range(0, nc)]
    for i in range(0, len(x)):
        centers[idxs[i]] += x[i]
        sizes[idxs[i]] += 1
    for i in range(0, nc):
        if sizes[i] > 0:
           centers[i] /= sizes[i]
        else:
           centers[i] = np.array(rnd.choice(x))
    # Считаем, насколько сместились (в сумме) центры кластеров
    delta = 0
    for i in range(0, nc):
        delta += pow(get_dist(old[i], centers[i]), 2)
  return idxs

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

def approx_trigo_poly(x0, xx, yz): # Частичная аппроксимация простым геометрическим полиномом
    def goal(a):
        r = yz - trigo_interp(a,xx)
        result = np.sum(r*r)
        return result
    def jacob(a):
        y0 = goal(a)
        result = np.array([0.0 for i in range(0, nump)])
        for i in range(0, nump):
            a[i] += dv
            y1 = goal(a)
            a[i] -= dv
            result[i] = (y1 - y0)/dv
        return result
    return minimize(goal, x0, method='Powell')

xd, yd, xv, yv, mins, maxs = load_data_wrapper()
idxs = clusterize_kmeans(xd, n_clusters)

nkoeffs = 0
qt = []
yt = []
kt = []
bt = []
ct = []
ir = []
for clust in range(0,n_clusters): # Перебираем все кластеры
# Для каждого кластера проводим упрощенную интерполяцию тригонометрическим полиномом
    x = []
    ys = []
    for i in range(0, len(xd)): # Выбираем данные текущего кластера
        if idxs[i] == clust:
           x.append(xd[i])
           ys.append(yd[i])
    nc = len(x)
    x = np.array(x)
    y = np.array(ys)
    yz = cp.copy(y).reshape((nc,1)) # Целевой вектор
    c = [np.zeros(nump) for i in range(0, xd[0].size)] # Матрица коэффициентов регрессионной функции
    k = [0.5 for i in range(0, xd[0].size+1)] # Масштабные множители
    b = [-1 for i in range(0, xd[0].size+1)] # Смещения
    y0 = np.zeros((nc,1))
    yp = 1000000 # Предыдущий прирост приближения
    for i in range(0, xd[0].size): # Цикл по входным переменным. На каждом шаге очередным геометрическим
# полиномом приближаем остаток, не учтенный предыдущими полиномами по предыдущим переменным
        xx = x[:,i]
        x0 = np.random.rand(nump)
        c[i] = approx_trigo_poly(x0, xx, yz).x
        if (abs(k[i]) < EPS):
           break
        y0 += 0.5*(trigo_interp(c[i], xx) + 1.0)/k[i] + b[i]
        d = y.reshape((nc,1)) - y0 # Вычисляем неучтенный остаток -- его попытаемся приблизить
                      # тригонометрическим полиномом по следующим переменным
        dd = np.sum(d*d)
        if dd > 1.1*yp: # Защищаемся от внезапного роста погрешности
           break
        yp = dd
        # Масштабируем остаток к диапазону [-1; 1]
        b[i+1] = np.min(d)
        k[i+1] = np.max(d)-np.min(d)
        if abs(k[i+1]) < EPS: # Если остаток мал, то можно ограничиться уже найденным фрагментом полинома
           break # Тогда завершаем обработку кластера
        else:
           k[i+1] = 1.0/k[i+1]
        yz = 2.0*k[i+1]*(d - b[i+1]) - 1.0
        yz = yz.reshape((nc,1))
        ireq = i # Запоминаем, сколько входных переменных требуют учета в текущем кластере
    nkoeffs += (nump+2)*ireq
    q = y.reshape((nc,1)) - full_interp(c,ireq,k,b,x)
    yt.append(ys)
    kt.extend(k)
    bt.extend(b)
    ct.append(c)
    ir.append(ireq)
    for i in range(0, nc):
        qt.append(q[i,0])
    print(clust, ". Средняя ошибка кластера = ", np.mean(np.abs(q)))
delta = np.abs(np.array(qt))
print("Вычислено коэффициентов: ", nkoeffs)
print("Средняя ошибка: ", np.mean(delta))
# Сохраняем данные обученного алгоритма
with open("trainedIDXS." + str(dest) + ".dat", 'wb') as f:
    pickle.dump(idxs,f) 
with open("trainedK." + str(dest) + ".dat", 'wb') as f:
    pickle.dump(kt,f) 
with open("trainedB." + str(dest) + ".dat", 'wb') as f:
    pickle.dump(bt,f) 
with open("trainedC." + str(dest) + ".dat", 'wb') as f:
    pickle.dump(ct,f) 
with open("trainedIR." + str(dest) + ".dat", 'wb') as f:
    pickle.dump(ir,f) 

print("Проверка по тестовой выборке:", end='')
err = np.array([[0.0]])
for i in range(0, len(xv)):
    x = xv[i]
    # Методом k-ближайшего соседа находим нужный кластер
    dists = [get_dist(x, xd[i]) for i in range(0, len(xd))] # Находим расстояния до всех образцовых точек
    votes = [0 for i in range(0, n_clusters)] # Голоса, поданные за кластеры
    for m in range(0,km):
        min_i = 0
        for r in range(1, len(xd)):
            if dists[r] < dists[min_i]:
               min_i = r
        votes[idxs[min_i]] += 1
        dists[min_i] = 10000000 # Чтобы больше не считать этого найденного соседа ближайшим
    max_vote = max(votes)
    # Ищем кластер с наибольшим числом голосов "За"
    for c in range(0, n_clusters):
        if votes[c] == max_vote:
           clust = c
           break
    base = clust*(xd[0].size+1)
    predicted = full_interp(ct[clust],ir[clust],kt[base:base+xd[0].size+1],bt[base:base+xd[0].size+1],x.reshape((1,10)))
    err += np.abs(yv[i]-predicted)
err /= len(xv)
print(err)
