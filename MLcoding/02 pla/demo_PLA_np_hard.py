# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 23:02:40 2018

@author: an
"""

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

test = csv.reader(open("data2.csv"))
data = np.ones([100,4])
ii = 0
for valu in test:
    data[ii,1:] = valu
    ii = ii+1
print(data)

# find w, b
w = np.random.randn(3)
flag = 1
w_all = np.random.randn(3) #搜集权重
error_num_all = [data.shape[0]] #搜集误差数量
change = 0
random_list = random.sample(list(np.arange(100)),100)
#基本思路与线性可分接近，不同在于将外循环的while改成for循环，内部增加了对w和误差数的搜集
for kk in np.arange(3):
    for jj in np.arange(100):
        ii = random_list[jj]
        print(ii)
        cal_res = np.sign(np.dot(w,data[ii,0:3]))
        if (cal_res != data[ii,-1]):
            w = data[ii,0:3]*data[ii,-1]+w  #更新权重w
            w_all = np.vstack((w_all,w))
            res = (data[:,-1] == np.sign(np.dot(data[:,0:3],w)))
            error_num = data.shape[0]-sum(res) # 统计误差个数
            error_num_all.append(error_num)
            continue
        elif (cal_res == data[ii,-1]) & (jj < len(random_list)-1): 
            continue
        elif (cal_res == data[ii,-1]) & (jj == len(random_list)-1) & (change == 0):
            flag = 0
            break

min_error_location = (np.where(error_num_all==np.min(error_num_all)))
w_final = w_all[min_error_location,:]
w = w_final[0][0]
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.linspace(4,8,50)
y = (-w[0]-w[1]*x)/w[2]
ax.scatter(data[0:50,1],data[0:50,2], c='b')
ax.scatter(data[50:100,1],data[50:100,2], c='r')
ax.plot(x,y,c='g')
#ax.scatter(data[:,1],data[:,2],30.0*(data[:,3]), 15.0*np.array(data[:,3]))
plt.show()