# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 22:11:37 2018

@author: Administrator
"""

from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]





def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return [frozenset(temp) for temp in C1]  ############### 修改书中map函数使用frozenset函数
                           

def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can in ssCnt:  ############### 修改书中has_key函数
                    ssCnt[can] += 1
                else: 
                    ssCnt[can] = 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems  ########## 计算每个元素的支持度
        if support >= minSupport:
            retList.insert(0,key)      ########## 总是把项集放到第一位
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk,k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort();         ########## 注意这里排序是必需的，这样可以从小到大进行组合，减少计算量
            L2.sort();
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport):
    C1 = createC1(dataSet)
    D = dataSet
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

def calcConf(freqSet, H, supportData, brl, minConf=0.7):  ##判断可信度
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            #print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7): # 循环调用，产生规则集合
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

dataset = loadDataSet()
retList, supportData = apriori(dataset, minSupport = 0.5)
print(retList)
print(supportData)

rules = generateRules(retList, supportData, minConf=0.7)
print('#############################################')
print(rules)
############################################################################
#################                  毒蘑菇案例               #################

mushData = [line.split() for line in open('mushroom.dat').readlines()]
L, supportData = apriori(mushData, minSupport=0.3)

for item in L[1]:
    if item.intersection('2'):
        print(item)


