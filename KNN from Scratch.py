# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Adham Erlangga Siwi)
"""
import numpy as np
import pandas as pd
from collections import Counter

#normalisasi data tiap kolom
def normAllCol(x):
	minmax = list()
	for i in range(len(x[0])):
		nilaiKolom = [row[i] for row in x]
		nilaiTerkecil = min(nilaiKolom)
		nilaiTerbesar = max(nilaiKolom)
		minmax.append([nilaiTerkecil, nilaiTerbesar])
	return minmax
 
#normalisasi seluruh data
def normAllData(x, minmax):
	for row in x:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
#perhitungan jarak dengan euclidean
def jarak(x1,x2):
    return np.sqrt(np.sum(x1-x2)**2)

#klasifikasi seluruh data test data
def predictAll(x):
    predicted_labels = [predict(x) for x in x]
    return np.array(predicted_labels)

#klasifikasi 1 data test data
def predict(x):
    distances = [jarak(x,xtrain) for xtrain in xtrain]
    indeksTetangga = np.argsort(distances)[:k] 
    labelsTetangga = [ytrain[i] for i in indeksTetangga]
    klasifikasi = Counter(labelsTetangga).most_common(1)
    return klasifikasi[0][0]

df = pd.read_csv (r'C:\Code\Kuliah\Pengantar AI\Tupro3\Diabetes.csv')

#split dataset
x = df.iloc[:, 0:8].values
y = df.iloc[:, 8].values

acc = []
accAvg = []
for i in range (1,21):
    k = i
    for j in range(1, 6):
        if j == 1:
            xtrain,xtest,ytrain,ytest = x[:615], x[614:], y[:615], y[614:]
            
            minmax = normAllCol(xtrain)
            normAllData(xtrain, minmax)
            minmax = normAllCol(xtest)
            normAllData(xtest, minmax)
            
            predictions = predictAll(xtest)
            akurasi = np.sum(predictions == ytest) / len (ytest)
            acc.append(akurasi)
            
        elif j == 2:
            xtrain =  np.concatenate([x[:462], x[614:]])
            xtest= np.concatenate([x[461:], x[:615]])
            ytrain= np.concatenate([y[:462], y[614:]])
            ytest = np.concatenate([y[461:], y[:615]])
            
            minmax = normAllCol(xtrain)
            normAllData(xtrain, minmax)
            minmax = normAllCol(xtest)
            normAllData(xtest, minmax)
            
            predictions = predictAll(xtest)
            akurasi = np.sum(predictions == ytest) / len (ytest)
            acc.append(akurasi)
            
        elif j == 3:
            xtrain =  np.concatenate([x[:308], x[461:]])
            xtest= np.concatenate([x[307:], x[:462]])
            ytrain= np.concatenate([y[:308], y[461:]])
            ytest = np.concatenate([y[307:], y[:462]])
            
            minmax = normAllCol(xtrain)
            normAllData(xtrain, minmax)
            minmax = normAllCol(xtest)
            normAllData(xtest, minmax)
            
            predictions = predictAll(xtest)
            akurasi = np.sum(predictions == ytest) / len (ytest)
            acc.append(akurasi)
            
        elif j == 4:
            xtrain =  np.concatenate([x[:155], x[307:]])
            xtest= np.concatenate([x[154:], x[:308]])
            ytrain= np.concatenate([y[:155], y[307:]])
            ytest = np.concatenate([y[154:], y[:308]])
            
            minmax = normAllCol(xtrain)
            normAllData(xtrain, minmax)
            minmax = normAllCol(xtest)
            normAllData(xtest, minmax)
            
            predictions = predictAll(xtest)
            akurasi = np.sum(predictions == ytest) / len (ytest)
            acc.append(akurasi)

        elif j == 5:
            xtrain,xtest,ytrain,ytest = x[154:], x[:155], y[154:], y[:155]
            
            minmax = normAllCol(xtrain)
            normAllData(xtrain, minmax)
            minmax = normAllCol(xtest)
            normAllData(xtest, minmax)
            
            predictions = predictAll(xtest)
            akurasi = np.sum(predictions == ytest) / len (ytest)
            acc.append(akurasi)

    accKNN = sum(acc) / len(acc)
    accAvg.append(accKNN)
    acc.clear()
    print("Rata-Rata Akurasi 5-foldCV pada K:",i ,"adalah",accKNN)
print("K terbaik:",accAvg.index(max(accAvg)) + 1,"dengan rata-rata akurasi", max(accAvg))    