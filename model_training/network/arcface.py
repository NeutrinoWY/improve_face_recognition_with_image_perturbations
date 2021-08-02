import torchvision.models as models
from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import time
import os
from network.resnet import resnet50
import torch.nn.functional as F
from torch.nn import Parameter




 
class Arcsoftmax(nn.Module):
    def __init__(self, feature_size, num_classes):
        super().__init__()
        self.w = Parameter(torch.randn((feature_size, num_classes)),requires_grad=True)   
        self.func = nn.Softmax()                                                        
 
    def forward(self, x, scale=64, margin=0.5):                                                  
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)                                            
 
        cosa = torch.matmul(x_norm, w_norm) / scale                                         
        a = torch.acos(cosa)
 
        arcsoftmax = torch.exp(scale * torch.cos(a + margin)) / (
                            torch.sum(torch.exp(scale * cosa), dim=1, keepdim=True) - torch.exp(
                            scale * cosa) + torch.exp(scale * torch.cos(a + margin)))                               
 
        return arcsoftmax
 
 
class ArcFace(nn.Module):
    def __init__(self, pretrained, feature_size, num_classes):
        super(ArcFace, self).__init__()
        
        self.sub_net = nn.Sequential(
            resnet50(pretrained=pretrained, feature_size=feature_size),                                                     
 
        )
        # when the last fc layer of resnet is not removed, add another fc as embedding layer
        #self.feature_net = nn.Sequential(
        #    nn.BatchNorm1d(1000),
        #    nn.LeakyReLU(0.1),                                                        
        #    nn.Linear(1000, feature_size, bias=False),
        #)
        self.arc_softmax = Arcsoftmax(feature_size=feature_size, num_classes=num_classes)                                         
 
    def forward(self, x):
        features = self.sub_net(x)
        y = torch.log(self.arc_softmax(features))                                                                                                       
        return features, y                                 
 
    def encode(self, x):
        return self.sub_net(x)                                      
 