# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:40:09 2018

@author: Administrator
"""

from numpy import *
from numpy import mat, linalg
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    data = loadtxt(fileName)    #用loadtxt函数方便
    dataMat = data[:,0:data.shape[1]-1]
    labelMat = data[:,data.shape[1]-1]  
    return dataMat,labelMat

def standRegres(dataMat,labelMat):
    xTx = (dataMat.T).dot(dataMat)
    xTy = (dataMat.T).dot(labelMat)
    xTx_inv = mat(xTx).I
    ws = xTx_inv.dot(xTy)
    return ws

dataMat,labelMat = loadDataSet("ex0.txt")
ws = standRegres(dataMat,labelMat)
print(ws)
m = dataMat.shape[0]
weights = np.eye((m))
testPoint = dataMat[0]
k = 1
for j in np.arange(m):
    diffMat = testPoint - dataMat[j,:]
    weights[j,j] = np.exp(diffMat.dot(diffMat.T/(-2*k**2)))
xTx = (dataMat.T).dot((weights.dot(dataMat)))
ws = np.array(mat(xTx).I).dot((dataMat.T).dot(weights.dot(labelMat)))
print(testPoint.dot(ws))

def lwlr(testPoint,dataMat,labelMat,k):
    m = dataMat.shape[0]
    weights = np.eye((m))
    for j in np.arange(m):
        diffMat = testPoint - dataMat[j,:]
        weights[j,j] = np.exp(diffMat.dot(diffMat.T/(-2*k**2)))
    xTx = (dataMat.T).dot((weights.dot(dataMat)))
    ws = np.array(mat(xTx).I).dot((dataMat.T).dot(weights.dot(labelMat)))
    #print(testPoint.dot(ws))
    return testPoint.dot(ws)
def testlwlr(testPoint,dataMat,labelMat,k):
    m = dataMat.shape[0]
    res = np.zeros(m)
    for i in np.arange(m):
        res[i] = lwlr(testPoint[i], dataMat, labelMat,k)
    return res
def rssError(yArr, yHatArr):
    return(np.linalg.norm(yArr-yHatArr, ord=2))

srtInd = dataMat[:,1].argsort(0)
xSort = dataMat[srtInd]

res = testlwlr(xSort,dataMat,labelMat,0.005)
plt.figure()
plt.scatter(dataMat[:,1], labelMat, c='red')
plt.plot(xSort[:,1],res, c = 'green')
plt.show()

dataMat,labelMat = loadDataSet("abalone.txt")
yHat01 = testlwlr(dataMat,dataMat,labelMat,0.1)
yHat1 = testlwlr(dataMat,dataMat,labelMat,1)
yHat10 = testlwlr(dataMat,dataMat,labelMat,10)

