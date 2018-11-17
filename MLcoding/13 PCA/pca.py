# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 20:53:56 2018

@author: Administrator
"""
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
#def loadDataSet(filename):
#    dataSet = loadtxt(filename)
#    return dataSet
def loadDataSet(fileName, delim = '\t'):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(delim)
        fltLine = []
        for i in curLine:
           fltLine.append(float(i))
        dataMat.append(fltLine)
    return np.array(dataMat)

def pca(dataMat, topN = 99):
    meanVal = mean(dataMat, axis = 0)
    meanRemoved = dataMat - meanVal
    convMat = cov(meanRemoved, rowvar = 0)
    eigVal, eigVec = linalg.eig(convMat)
    eigValInd = argsort(eigVal)
    eigValInd = eigValInd[:-(topN+1):-1]
    redEigVec = eigVec[:,eigValInd]
    lowDataMat = np.dot(meanRemoved,redEigVec)    
    reconMat = np.dot(lowDataMat,redEigVec.T) + meanVal
    return lowDataMat, reconMat

dataMat = loadDataSet('testSet.txt')
lowDataMat, reconMat = pca(dataMat, 1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0],dataMat[:,1], marker='^')
ax.scatter(reconMat[:,0], reconMat[:,1], marker = 'o', c = 'red')
plt.show()


datamat = loadDataSet('secom.data',' ');
my_imputer = Imputer()
data_imputed = my_imputer.fit_transform(datamat)

meanVal = mean(data_imputed, axis = 0)
meanRemoved = data_imputed - meanVal
convMat = cov(meanRemoved, rowvar = 0)
eigVal, eigVec = linalg.eig(convMat)
eig_sort = sort(eigVal)
eig_sort_invert = eig_sort[:-1:-1]


    
    