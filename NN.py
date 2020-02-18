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
def enc(i, num):
    return np.array([i >> d & 1 for d in range(num)])


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



trans_img = transforms.Compose([transforms.ToTensor()])
train_x = torchvision.datasets.FashionMNIST("./data/", train=True, transform=trans_img, download=True)
test_x = torchvision.datasets.FashionMNIST("./data/", train=False, transform=trans_img, download=True)
trainloader = torch.utils.data.DataLoader(train_x, batch_size=512, shuffle = True)
testloader = torch.utils.data.DataLoader(test_x, batch_size=len(test_x), shuffle = True)


mod = Model(784,10,128)


loss1 = nn.CrossEntropyLoss()
opt = torch.optim.Adam(mod.parameters(), lr=0.001)

for i in tqdm(range(50)):
    mod.train()
    ij=0
    
    train_correct=0
    corr=0
    tl=0
    for i, (x, y) in enumerate(trainloader):
        x = Variable(x)
        y = Variable(y)
        opt.zero_grad()
        
        ot = mod(x)
        l=loss1(ot,y)
        l.backward()
        opt.step()
        pred = torch.max(ot.data,1)
        
        corr +=(pred[1] == y).sum().item()
        ij+=1
    
    tracc=corr/len(train_x)
    test_correct=0
    print("*********************************",len(train_x),tl)
    tl=0
    for i, (x, y) in enumerate(testloader):
        x = Variable(x)
        target = Variable(y)
        ot = mod(x)
        pred = torch.max(ot.data,1)
        
        test_correct +=(pred[1] == y).sum().item()
    tsacc=test_correct/len(test_x)
    print("*********************************",len(test_x),tl)
    print("Train Accuracy :"+str(tracc)+", Test Accuracy :"+str(tsacc))
torch.save(mod, "model/NN.pth")
# torch.save(model.state_dict(), "models/convNet.pt")
