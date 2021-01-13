#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 14:32:35 2020

@author: yan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self, encoder_arch, decoder_arch):
        super(Net, self).__init__()        
        self.encoder = self.Encoder(encoder_arch)
        self.decoder = self.Decoder(decoder_arch)
        
    def Encoder(self, encoder_arch):
        modules = []
        for i in range(len(encoder_arch)-1):
            modules.append(torch.nn.Linear(encoder_arch[i][0],encoder_arch[i][1]))
            modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(encoder_arch[-1][0],encoder_arch[-1][1]))
        modules.append(torch.nn.Tanh())
        encoder = nn.Sequential(*modules)        
        return encoder
        
    def Decoder(self, decoder_arch):
        modules = []
        for i in range(len(decoder_arch)-1):
            modules.append(torch.nn.Linear(decoder_arch[i][0],decoder_arch[i][1]))
            modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(decoder_arch[-1][0],decoder_arch[-1][1]))
        modules.append(torch.nn.Tanh())
        decoder = nn.Sequential(*modules) 
        return decoder
    
    def forward(self,x):
        y = self.encoder.forward(x)
        tmp = torch.cat((x,y),0)
        z = self.decoder.forward(tmp)        
        return y,z
    
    def forward(self,x,c):
        y = self.encoder.forward(x)
        tmp = torch.cat((x,c),0)
        z = self.decoder.forward(tmp)        
        return y,z
    
class ExoskeletonAENet(nn.Module):    
    def __init__(self, encoder_arch):
        super().__init__()  
        modules = []        
        for i in range(len(encoder_arch)-1):
            modules.append(torch.nn.Linear(encoder_arch[i][0],encoder_arch[i][1]))
            modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(encoder_arch[-1][0],encoder_arch[-1][1]))
#        for item in encoder_arch:
#            modules.append(torch.nn.Linear(item[0],item[1]))
#            modules.append(torch.nn.ReLU())
#        modules.append(torch.nn.Linear(item[1],6))
        modules.append(torch.nn.Tanh())
        self.net = nn.Sequential(*modules)     
        
    def forward(self, x):
        return self.net.forward(x)  

        