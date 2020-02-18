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
import torch
import torchvision
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
from utils import AverageMeter

print("NAME:  MANASVI AGGARWAL")
print("DEPARTMENT:  CSA")
print("COURSE:  MTECH(RES.)")
print("SR NO:  16223")

def enc(i, num):
    return np.array([i >> d & 1 for d in range(num)])


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=8, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # self.cnn3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding=0)
        # self.relu3 = nn.ReLU()
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        # self.relu2 = nn.ReLU()
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # self.siz=0
        # self.dropout = nn.Dropout(p=0.5)
        self.adapt = nn.AdaptiveMaxPool2d((5,5))
        
        
        self.fc1 = nn.Linear(64*5*5,512)
        self.fc2 = nn.Linear(512, 192)
        self.fc3 = nn.Linear(192, 10)
    
    def forward(self, x):
        
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        
        out = self.cnn2(out)
        out = self.relu2(out)
        
        out = self.maxpool2(out)
        out=self.adapt(out)
        
        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)
        sz=list(out.size())
        sz=sz[-1]
        # out = self.dropout(out)

        
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out


class Model(nn.Module):
    def __init__(self, numm, classes, sze):
        super(Model,self).__init__()
        layr = []   
        layr.append(nn.Linear(numm,512))
        layr.append(nn.ReLU())

        layr.append(nn.Linear(512,128))
        # layr.append(nn.BatchNorm1d(sze))
        layr.append(nn.ReLU())

        layr.append(nn.Linear(128,16))
        # layr.append(nn.BatchNorm1d(sze))
        layr.append(nn.ReLU())

        
        self.lays = nn.Sequential(*layr)
        self.outp = nn.Linear(16, classes)

    def forward(self,x):
        # print(list(x.size()))
        x = x.view(-1, 784)
        # print(list(x.size()))
        x1 = self.lays(x)
        out = self.outp(x1)   
        return out 

#part 1
def test(model2,testloader):
    model2.eval()
    corr = 0
    tl = 0
    y_gt = []
    y_pred_label = []
    avg_loss = AverageMeter("average-loss")

    for x, y in testloader:
        x = Variable(x)
        # print(list(x.size()))
        ot = model2(x)
        y_pred = F.softmax(ot, dim=1)
        y_pred_label_tmp = torch.argmax(y_pred, dim=1)
        loss = F.cross_entropy(ot, y)
        avg_loss.update(loss, x.shape[0])

        y_gt += list(y.numpy())
        y_pred_label += list(y_pred_label_tmp.numpy())
        
    return avg_loss.avg, y_gt, y_pred_label

    # print("*******************",tl,len(test_x))
    print(' Accuracy train: {}. Accuracy test: {}'.format(acc_tr, acc_ts))

if __name__ == "__main__":
    trans_img = transforms.Compose([transforms.ToTensor()])
    train_x = torchvision.datasets.FashionMNIST("./data/", train=True, transform=trans_img, download=True)
    test_x = torchvision.datasets.FashionMNIST("./data/", train=False, transform=trans_img, download=True)
    trainloader = torch.utils.data.DataLoader(train_x, batch_size=400, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_x, batch_size=512, shuffle = True)
    model2 = torch.load("model/NN.pth")
    model2.eval()

 
    loss, gt, pred = test(model2, testloader)
    print("ACC",np.mean(np.array(gt) == np.array(pred)))
    with open("multi-layer-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))


    model2 = torch.load("model/CNN.pth")
    loss, gt, pred = test(model2, testloader)
    print("ACC",np.mean(np.array(gt) == np.array(pred)))
    with open("convolution-neural-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))


