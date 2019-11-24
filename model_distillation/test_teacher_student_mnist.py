# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:39:55 2019

@author: Administrator
"""

# coding: utf-8

import torch
from torch.utils.data import DataLoader
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
#from Code.utils.utils import MyDataset, validate, show_confMat
from tensorboardX import SummaryWriter
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data



class anNet(nn.Module):
    def __init__(self):
        super(anNet,self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.pool1 = nn.MaxPool2d(2,1)
        self.fc3 = nn.Linear(3750,10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(F.relu(x))
        x = x.view(x.size()[0],-1)
        x = self.fc3(x)
        return x
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


class anNet_deep(nn.Module):
    def __init__(self):
        super(anNet_deep,self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1,64,3,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU())
        self.conv2 = nn.Sequential(
                nn.Conv2d(64,64,3,1,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU())
        self.conv3 = nn.Sequential(
                nn.Conv2d(64,128,3,1,padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU())
        self.conv4 = nn.Sequential(
                nn.Conv2d(128,128,3,1,padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU())
        self.conv5 = nn.Sequential(
                nn.Conv2d(128,256,3,1,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU())
        self.pooling1 = nn.Sequential(nn.MaxPool2d(2,stride=2))
        self.fc = nn.Sequential(nn.Linear(6272,10))
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pooling1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pooling1(x)
        x = x.view(x.size()[0],-1)
        x = self.fc(x)
        return x
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

correct_ratio = []
alpha = 0.5

img_rows = 28
img_cols = 28
MNIST_data_folder="E:\\study\\MNIST_data"
mnist = input_data.read_data_sets(MNIST_data_folder,one_hot = False)
X_train, y_train = mnist.train.images,mnist.train.labels
X_test, y_test = mnist.test.images, mnist.test.labels
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255.0
X_test = X_test/255.0

torch_dataset_train = Data.TensorDataset(torch.from_numpy(np.double(X_train)),
                                         torch.from_numpy(np.int64(y_train)))
torch_dataset_test = Data.TensorDataset(torch.from_numpy(np.double(X_test)),
                                         torch.from_numpy(np.int64(y_test)))
trainload = torch.utils.data.DataLoader(
        torch_dataset_train,
        batch_size = 64,
        shuffle = True)
testload = torch.utils.data.DataLoader(
        torch_dataset_test,
        batch_size = 512,
        shuffle = False)
teach_model = anNet_deep()
teach_model.cuda()
teach_model.load_state_dict(torch.load('teach_net_params_0.9895.pkl'))

model = anNet()
model.cuda()
#model.load_state_dict(torch.load('student_net_params_0.9598.pkl'))
#model.load_state_dict(torch.load('teach_net_params_0.9895.pkl'))
criterion = nn.CrossEntropyLoss()
criterion2 = nn.KLDivLoss()

optimizer = optim.Adam(model.parameters(),lr = 0.001)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(200):
    loss_sigma = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(trainload):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        optimizer.zero_grad()
        
        outputs = model(inputs.float())
        loss1 = criterion(outputs, labels)
        
        teacher_outputs = teach_model(inputs.float())
        T = 2
        outputs_S = F.softmax(outputs/T,dim=1)
        outputs_T = F.softmax(teacher_outputs/T,dim=1)
        loss2 = criterion2(outputs_S,outputs_T)*T*T
        
        loss = loss1*(1-alpha) + loss2*alpha

#        loss = loss1
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, dim = 1)
        total += labels.size(0)
        correct += (predicted.cpu()==labels.cpu()).squeeze().sum().numpy()
        loss_sigma += loss.item()
        if i% 100 == 0:
            loss_avg = loss_sigma/10
            loss_sigma = 0.0
            print('loss_avg:{:.2}   Acc:{:.2%}'.format(loss_avg, correct/total))
    if epoch % 2 == 0:
        loss_sigma = 0.0
        cls_num = 10
        conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
        model.eval()
        for i, data in enumerate(testload):

            # 获取图片和标签
            images, labels = data
            images, labels = Variable(images), Variable(labels)
            images = images.cuda()
            labels = labels.cuda()
            # forward
            outputs = model(images.float())
            outputs.detach_()
            
            # 计算loss
            loss = criterion(outputs, labels)
            loss_sigma += loss.item()

            # 统计
            _, predicted = torch.max(outputs.data, 1)
            # labels = labels.data    # Variable --> tensor

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels.cpu()[j].numpy()
                pre_i = predicted.cpu()[j].numpy()
                conf_mat[cate_i, pre_i] += 1.0
        net_save_path = 'student_net_params_' + str(np.around(conf_mat.trace()/conf_mat.sum(),decimals = 4)) + '.pkl'
        torch.save(model.state_dict(),net_save_path)
        print('-------------------------{} set Accuracy:{:.4%}---------------------'.format('Valid', conf_mat.trace() / conf_mat.sum()))
