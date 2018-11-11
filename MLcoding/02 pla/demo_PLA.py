# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 22:30:38 2018

@author: an
"""

import csv
import numpy as np
import random
import matplotlib.pyplot as plt
#test = pd.read_csv("data1.csv")

test = csv.reader(open("data1.csv"))
data = np.ones([100,4])
ii = 0
for valu in test:
    data[ii,1:] = valu
    ii = ii+1
print(data)

# find w, b
w = np.random.randn(3)
w_old = np.random.randn(3)

flag = 1
while(flag == 1):  ##两重循环，flag循环用于在更新权重时重新计算，jj循环用于对标签计算
    change = 0
    random_list = random.sample(list(np.arange(100)),100)
    for jj in np.arange(100):
        ii = random_list[jj]
        print(ii)
        cal_res = np.sign(np.dot(w,data[ii,0:3]))
        if (cal_res != data[ii,-1]):
            w = data[ii,0:3]*data[ii,-1]+w  #更新权重w
            change = 1
            break
        elif (cal_res == data[ii,-1]) & (jj < len(random_list)-1): 
            continue
        elif (cal_res == data[ii,-1]) & (jj == len(random_list)-1) & (change == 0):
            flag = 0
            break
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.linspace(4,7.5,50)
y = (-w[0]-w[1]*x)/w[2]
ax.scatter(data[0:50,1],data[0:50,2], c='b')
ax.scatter(data[50:100,1],data[50:100,2], c='r')
ax.plot(x,y,c='g')
#ax.scatter(data[:,1],data[:,2],30.0*(data[:,3]), 15.0*np.array(data[:,3]))
plt.show()