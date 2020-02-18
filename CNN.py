import torch
import torchvision
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import pandas as pd
import sys
import scipy as sc
import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.utils.data as Data
import pandas as pd
import sys
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=8, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.cnn4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.siz=0
        self.dropout = nn.Dropout(p=0.5)
        self.adapt = nn.AdaptiveMaxPool2d((5,5))
        
        
        self.fc1 = nn.Linear(576,512)
        self.fc2 = nn.Linear(512, 192)
        self.fc3 = nn.Linear(192, 10)
    
    def forward(self, x):
        
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        
        out = self.cnn2(out)
        out = self.relu2(out)
        
        out = self.maxpool2(out)

        # out = self.cnn3(out)
        # out = self.relu3(out)
        
        # out = self.maxpool3(out)

        out = out.view(out.size(0), -1)
        sz=list(out.size())
        sz=sz[-1]
        # print("*******************",sz)        

        # out = self.cnn4(out)
        # out = self.relu4(out)
        
        # out = self.maxpool4(out)
        # out=self.adapt(out)
        
        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)

        # out = self.dropout(out)

        
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out


trans_img = transforms.Compose([transforms.ToTensor()])
train_x = torchvision.datasets.FashionMNIST("./data/", train=True, transform=trans_img, download=True)
test_x = torchvision.datasets.FashionMNIST("./data/", train=False, transform=trans_img, download=True)
trainloader = torch.utils.data.DataLoader(train_x, batch_size=512, shuffle = True)
testloader = torch.utils.data.DataLoader(test_x, batch_size=len(test_x), shuffle = True)

mod = CNN()

loss1 = nn.CrossEntropyLoss()
opt = torch.optim.Adam(mod.parameters(), lr=0.001)

iter = 0
for epoch in range(25):
    corr=0
    tl=0
    train_total=0
    for i, (x, y) in enumerate(trainloader):
        # print(list(x.size()))
        x = Variable(x)
        y = Variable(y)
        opt.zero_grad()
        ot = mod(x)
        l = loss1(ot, y)
        l.backward()
        opt.step()
        # count=count+1
        # train_total=train_total+y.size(0)
        _, pr = torch.max(ot.data, 1)
        tl += y.size(0)
        corr += (pr == y).sum()
    acc_tr = 100 * corr / len(train_x)
    # print("*******************",tl,len(train_x))
    corr = 0
    tl = 0
    for x, y in testloader:
        x = Variable(x)
        ot = mod(x)
        _, pr = torch.max(ot.data, 1)                                           
        # tl += y.size(0)
        corr += (pr == y).sum()
    acc_ts = 100 * corr / len(test_x)
    # print("*******************",tl,len(test_x))
    print(' Accuracy train: {}. Accuracy test: {}'.format(acc_tr, acc_ts))

torch.save(mod, "model/CNN.pth")
print()
