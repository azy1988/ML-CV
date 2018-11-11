# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 23:38:58 2018

@author: Administrator
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


fr = open('datingTestSet2.txt','rb')
numberOfLines = len(fr.readlines())
returnMat = np.zeros((numberOfLines, 3)) 
classLabelVector = [] # 存放标签
fr = open('datingTestSet2.txt','rb')
index = 0


#def clssify0(testData, labeledData, labels, 3):
    


for line in fr.readlines():
    line = line.strip().decode()
    #print(line)
    #line = str(line).strip('b\'')
    listFromLine = line.split('\t')
    returnMat[index, :] = listFromLine[0:3]
    classLabelVector.append(int(listFromLine[3]))
    index = index+1
#return classLabelVector, returnMat

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(returnMat[:,0],returnMat[:,1],15.0*np.array(classLabelVector), 15.0*np.array(classLabelVector))
plt.show()

minVals = returnMat.min(0)
maxVals = returnMat.max(0)
denominator = maxVals - minVals
returnMat_norm = (returnMat - np.tile(minVals,(index,1)))/np.tile(denominator,(index,1))
hratio = 0.1
numTestVecs = int(index * hratio)
right_num = 0
#for i in range(numTestVecs):
#    classifierResult = classify0(returnMat_norm[i, :], returnMat_norm[numTestVecs:index, :], classLabelVector[numTestVecs:index], 3)
#    if classifierResult ==  classLabelVector[i]:
#        right_num = right_num + 1
inx = returnMat_norm[0:numTestVecs,:]
label_dataSet = returnMat_norm[numTestVecs:,:]
labels = classLabelVector[numTestVecs:]


test_labels = classLabelVector[0:numTestVecs]

k = 3
def classify0(inx, label_dataSet, labels, k):
    [m,n] = np.shape(label_dataSet)
    [num,n] = np.shape(inx)
    cal_label = np.zeros(num)
    for ii in np.arange(num):
        inx_cur = inx[ii,:]
        inx_cur_copy = np.tile(inx_cur,[m,1])
        distance = np.sqrt(np.sum(np.square((inx_cur_copy-label_dataSet)),1))
        sortedDistIndicies = distance.argsort()
        class_num = [0,0,0]
        for jj in np.arange(k):
            cur_label = sortedDistIndicies[jj]
            class_num[labels[cur_label]-1] =  class_num[labels[cur_label]-1] + 1
        cal_label[ii] = class_num.index(max(class_num)) + 1
    return cal_label


cal_label = classify0(inx, label_dataSet, labels, k)
right_num = np.sum((test_labels-cal_label) == 0)
print("识别正确率为 %.2f %% " % (right_num/numTestVecs*100))


-0.2087751808847349
-0.23734723774477834
-0.26856659830640084
-0.27425740924791686
他代表的是对象属性与对象值之间的一种映射关系。树中每个节点表示某个对象，而每个分叉路径则代表的某个可能的属性值，而每个叶结点则对应从根节点到该叶节点所经历的路径所表示的对象的值。