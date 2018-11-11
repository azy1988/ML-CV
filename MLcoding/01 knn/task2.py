# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 23:59:24 2018

@author: Administrator
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import operator

#def img2vector(filename):
#    returnVect = np.zeros((1,1024))
#    fr = open(filename)
#    for i in range(32):
#        lineStr = fr.readline()
#        for j in range(32):
#            returnVect[0,32*i+j] = int(lineStr[j])
#    return returnVect


def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    all_vect = fr.readlines()
    for i in np.arange(32):
        for j in np.arange(32):
                returnVect[0,32*i+j] = all_vect[i][j]
    return returnVect

def classify0(inx, label_dataSet, labels, k):
    [m,n] = np.shape(label_dataSet)
    [num,n] = np.shape(inx)
    cal_label = np.zeros(num)
    for ii in np.arange(num):
        inx_cur = inx[ii,:]
        inx_cur_copy = np.tile(inx_cur,[m,1])
        distance = np.sqrt(np.sum(np.square((inx_cur_copy-label_dataSet)),1))
        sortedDistIndicies = distance.argsort()
        class_num1 = np.zeros((1,max(labels)+1))
        class_num = class_num1.tolist()[0]
        for jj in np.arange(k):
            cur_label = sortedDistIndicies[jj]
            class_num[labels[cur_label]] =  class_num[labels[cur_label]] + 1
        cal_label[ii] = class_num.index(max(class_num)) 
    return cal_label


def handwritingClassTest():
    # 1. 导入训练数据
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 将 32*32的矩阵->1*1024的矩阵
        print('trainingDigits/%s' % fileNameStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    # 2. 导入测试数据
    testFileList = os.listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount / float(mTest)))
    
handwritingClassTest()
    