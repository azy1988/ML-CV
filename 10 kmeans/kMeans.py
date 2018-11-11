'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''
from numpy import *
from urllib import parse
from urllib import request
import numpy as np
from urllib.request import urlopen
import matplotlib
import matplotlib.pyplot as plt
import urllib
import json
from time import sleep
import urllib
##导入数据
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = np.loadtxt(fileName)
    return dataMat

##计算距离
def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2))) #la.norm(vecA-vecB)

##随机初始化聚点
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] =np.mat(minJ + rangeJ * np.random.rand(k,1))
    return centroids

##kmeans
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2))) #保存数据类别与距离 
    centroids = createCent(dataSet, k)       #保存聚类点  
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m): # 判断每个点距离所属类别是否最近
            minDist = np.inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2  ##记录下第i个样本所属类别，距离
        print(centroids)
        for cent in range(k):    # 重新计算距离，聚类中心点
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust, axis=0) 
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] 
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1]) 
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) 
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
    return mat(centList), clusterAssment

dataSet = loadDataSet('testSet.txt')
centroids, clusterAssment = biKmeans(dataSet, 4)
plt.figure()
plt.scatter(dataSet[:,0],dataSet[:,1])
plt.scatter(centroids[:,0].flatten().A[0],centroids[:,1].flatten().A[0],marker='+', s=300)
plt.show()


#def massPlaceFind(fileName):
#    fw = open('places.txt', 'w')
#    for line in open(fileName).readlines():
#        line = line.strip()
#        lineArr = line.split('\t')
#        retDict = geoGrab(lineArr[1], lineArr[2])
#        if retDict['ResultSet']['Error'] == 0:
#            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
#            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
#            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
#            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
#        else: print("error fetching")
#        sleep(1)
#    fw.close()
#    
#def distSLC(vecA, vecB):#Spherical Law of Cosines
#    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
#    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
#                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
#    return arccos(a + b)*6371.0 #pi is imported with numpy
#def geoGrab(stAddress, city):
#    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
#    params = {}
#    params['flags'] = 'J'#JSON return type
#    params['appid'] = 'eleTFd7a'
#    params['location'] = '%s %s' % (stAddress, city)
#    url_params = parse.urlencode(params)
#    yahooApi = apiStem + url_params      #print url_params
#    print(yahooApi)
#    c=request.urlopen(yahooApi)
#    return json.loads(c.read())
#
#def clusterClubs(numClust=5):
#    datList = []
#    for line in open('places.txt').readlines():
#        lineArr = line.split('\t')
#        datList.append([float(lineArr[4]), float(lineArr[3])])
#    datMat = mat(datList)
#    myCentroids, clustAssing = kMeans(datMat, numClust, distMeas=distSLC)
#    fig = plt.figure()
#    rect=[0.1,0.1,0.8,0.8]
#    scatterMarkers=['s', 'o', '^', '8', 'p', \
#                    'd', 'v', 'h', '>', '<']
#    axprops = dict(xticks=[], yticks=[])
#    ax0=fig.add_axes(rect, label='ax0', **axprops)
#    imgP = plt.imread('Portland.png')
#    ax0.imshow(imgP)
#    ax1=fig.add_axes(rect, label='ax1', frameon=False)
#    for i in range(numClust):
#        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
#        markerStyle = scatterMarkers[i % len(scatterMarkers)]
#        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
#    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
#    plt.show()
