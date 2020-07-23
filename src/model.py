import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
sys.path = [
    'C:\\Users\\pka\\panda_kaggle\\EfficientNet-PyTorch-master',
] + sys.path
from efficientnet_pytorch import model as eff_net

pre_train_eff = {
    'efficientnet-b0': 'efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'efficientnet-b7-dcc49843.pth'
    }

class Net(nn.Module):
    def __init__(self, name):
        super(Net, self).__init__()
        self.eff_net = eff_net.EfficientNet.from_name(name)
        self.eff_net.load_state_dict(torch.load(os.path.join(
                                                'C:\\Users\\pka\\kaggle\\EfficientNet (Standard Training & Advprop)',
                                                pre_train_eff[name])))        
        self.fc = nn.Linear(self.eff_net._fc.in_features, 5)       
        self.eff_net._fc = nn.Identity() 
    
    def current_net(self, x):
        return self.eff_net(x)
    
    def forward(self, x):
        x = self.current_net(x)
        x = self.fc(x)
        return x


class resnet32(nn.Module):
    def __init__(self, pretrained):
        super(resnet32, self).__init__()
        self.res_net = models.resnext50_32x4d(pretrained = pretrained)
        self.fc_ = nn.Linear(self.res_net.fc.out_features, 5)
    
    def forward(self, x):
        x = self.fc_(x)
        return x