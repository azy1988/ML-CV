# Check out the code in work at https://www.kaggle.com/hsankesara/prototypical-net/
# Check out the blog at <COMING SOON>

import torch
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm
import multiprocessing as mp
from preprocessing import read_images
from prototypicalNet import PrototypicalNet, train_step, test_step, load_weights
tqdm.pandas(desc="my bar!")


#def main():
    # Reading the data
trainx, trainy = read_images('./omniglot/images_background')
testx, testy = read_images('./omniglot/images_evaluation')
# Checking if GPU is available
print('finish read image')
use_gpu = torch.cuda.is_available()
use_gpu = False
# Converting input to pytorch Tensor
trainx = torch.from_numpy(trainx).float()
testx = torch.from_numpy(testx).float()
if use_gpu:
    trainx = trainx.cuda()
    testx = testx.cuda()
# Priniting the data
print(trainx.size(), testx.size())
# Set training iterations and display period
num_episode = 16000
frame_size = 100
trainx = trainx.permute(0, 3, 1, 2)
testx = testx.permute(0, 3, 1, 2)
# Initializing prototypical net
protonet = PrototypicalNet(use_gpu)

#pretrained_dict = torch.load('98.0400param.pkl')
#net_state_dict = protonet.state_dict()
#protonet.load_state_dict(net_state_dict)

print('net')
# Training loop
frame_loss = 0
frame_acc = 0
for i in range(num_episode):
    print(i)
    loss, acc = train_step(protonet, trainx, trainy, 5, 15, 5)
    frame_loss += loss.data
    frame_acc += acc.data
    if((i+1) % frame_size == 0):
        print("Frame Number:", (i), 'Frame Loss: ', frame_loss.data.cpu().numpy().tolist() /
              frame_size, 'Frame Accuracy:', (frame_acc.data.cpu().numpy().tolist() * 100) / frame_size)
        log_dir = '{:.4f}'.format((frame_acc.data.cpu().numpy().tolist() * 100) / frame_size) + 'param.pkl'
        torch.save(protonet,log_dir)
        frame_loss = 0
        frame_acc = 0

# Test loop
num_test_episode = 2000
avg_loss = 0
avg_acc = 0
for _ in range(num_test_episode):
    loss, acc = test_step(protonet, testx, testy, 5, 15, 15)
    avg_loss += loss.data
    avg_acc += acc.data
print('Avg Loss: ', avg_loss.data.cpu().numpy().tolist() / num_test_episode,
      'Avg Accuracy:', (avg_acc.data.cpu().numpy().tolist() * 100) / num_test_episode)

# Using Pretrained Model
protonet = load_weights('./protonet.pt', protonet, use_gpu)


#if __name__ == "__main__":
#    main()
