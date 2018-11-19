# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 20:43:46 2018

@author: Administrator
"""

from numpy import *
import numpy as np

U, sigma, VT = linalg.svd([[1,1],[7,7]])

sigma_all = np.eye(2)*sigma
print(np.dot(np.dot(U,sigma_all),VT))

def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = linalg.norm(inA)*linalg.norm(inB)
    return 0.5+0.5*(num/denom)

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
    
data = loadExData()
U, sigma, VT = linalg.svd(data)
sigma[2:5] = 0
a = list(sigma)
b = list(zeros(len(U)-len(VT)))
a.extend(b)
num = len(VT)
sigma_all = np.eye(len(U))*np.array(a)
rescon = U.dot(sigma_all[:,0:num]).dot(VT)
print(rescon)

#############################################################
#############################################################
# 协同过滤


def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: continue
        overLap = nonzero(logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0]
        if len(overLap) == 0: similarity = 0
        else: similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])
        print('第 %d 个人 %d 个特征 相似度: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
    
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #create transformed items
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print('第 %d 个人 %d 个特征 相似度: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems) == 0: return '此人已经全部投票'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

myMat = loadExData2()
recommend = recommend(mat(myMat), 2)
print(recommend)
#############################################################
#############################################################
# 图像压缩

def printMat(inMat, thresh = 0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k])>thresh:
                print(1, end = "")
            else: 
                print(0, end = "")
        print('')
thresh = 0.8
myl = []
for line in open('0_5.txt').readlines():
    newRow = []
    for i in range(32):
        newRow.append(int(line[i]))
    myl.append(newRow)
myMat = mat(myl)
print("-----------------原始图像-------------------")
printMat(myMat, thresh)
U,sigma, VT = linalg.svd(myMat)

sigma[2:-1] = 0
a = list(sigma)
b = list(zeros(len(U)-len(VT)))
a.extend(b)
num = len(VT)
sigma_all = np.eye(len(U))*np.array(a)
rescon = U.dot(sigma_all[:,0:num]).dot(VT)

print("-----------------压缩图像-------------------")
printMat(rescon, thresh)



