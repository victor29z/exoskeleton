#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 14:32:16 2020

@author: yan
"""

import exoskeleton_dataset
import network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


#encoder_arch = [[6,14],[14,28],[28,28],[28,6]]
#decoder_arch = [[12,14],[14,28],[28,28],[28,7]]
#net = network.Net(encoder_arch,decoder_arch)
#dataset = exoskeleton_dataset.ExoskeletonDataset(file="data/exoskeleton_data",root_dir="./")
#sample_d = dataset[0]
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#loss = nn.MSELoss()

def exoskeleton_ae_network():
    encoder_arch = [[6,14],[14,28],[28,28],[28,6]]
    decoder_arch = [[12,24],[24,28],[28,28],[28,7]]
    
    encoder = network.ExoskeletonAENet(encoder_arch)
    decoder = network.ExoskeletonAENet(decoder_arch)    
    return encoder, decoder

def train_s(encoder, decoder, loss_en, loss_de, optimizer_en,
          optimizer_de, train_dataset):    
    target = torch.Tensor()
    constrains = torch.Tensor()
    master = torch.Tensor()
        
    tt = exoskeleton_dataset.ToTensor()
    
    for item in train_dataset:        
        tensor_data = tt(item)
        tmp = tensor_data["target"].unsqueeze(dim=1)
        target = torch.cat((target, tmp), 1)
        tmp = tensor_data["constrains"].unsqueeze(dim=1)
        constrains = torch.cat((constrains, tmp), 1)
        tmp = tensor_data["master"].unsqueeze(dim=1)
        master = torch.cat((master, tmp), 1)
    
    optimizer_en.zero_grad()
    optimizer_de.zero_grad()

    x = encoder.forward(torch.transpose(target,0,1))
    tmp = torch.cat((target,constrains),0)
    y = decoder.forward(torch.transpose(tmp,0,1))
    
    le = loss_en.forward(torch.transpose(x,0,1), constrains)
    ld = loss_de.forward(torch.transpose(y,0,1),master)
    
    le.backward()
    ld.backward()
    
    optimizer_en.step()
    optimizer_de.step()    
    return le.item(), ld.item()

def train(model, loss, optimizer, train_dataset):   
    target = torch.Tensor()
    constrains = torch.Tensor()
    master = torch.Tensor()
        
    tt = exoskeleton_dataset.ToTensor()
    
    for item in train_dataset:        
        tensor_data = tt(item)
        tmp = tensor_data["target"].unsqueeze(dim=1)
        target = torch.cat((target, tmp), 1)
        tmp = tensor_data["constrains"].unsqueeze(dim=1)
        constrains = torch.cat((constrains, tmp), 1)
        tmp = tensor_data["master"].unsqueeze(dim=1)
        master = torch.cat((master, tmp), 1)
    
    optimizer.zero_grad()
    x = model.forward(torch.transpose(target,0,1))
    l = loss.forward(x,torch.transpose(master,0,1))
    
    l.backward()
    optimizer.step()
    return l.item()


def predict(encoder, target):    
    return encoder.forward(target)

def train_separately(train_dataset, val_dataset, test_dataset):    
    loss_en = nn.MSELoss()
    loss_de = nn.MSELoss()

    n_examples = len(train_dataset)  
    encoder, decoder = exoskeleton_ae_network()    
    
    optimizer_en = optim.SGD(encoder.parameters(), lr=0.01, momentum=0.9)
    optimizer_de = optim.SGD(decoder.parameters(), lr=0.01, momentum=0.9)    
    
    batch_size = 500
    num_batches = n_examples // batch_size

    for i in range(5000):
        cost_le = 0.
        cost_ld = 0.
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            le, ld = train_s(encoder, decoder, loss_en, loss_de, optimizer_en, 
                          optimizer_de, train_dataset[start: end])
            cost_le += le
            cost_ld += ld
            
        print("Epoch = {epoch}, cost_le = {le}, cost_ld = {ld}".format(
        epoch = i+1, le = cost_le / num_batches,ld = cost_ld / num_batches))
#        print ("Epoch %d, cost_le = %f, cost_ld = %f", i + 1, 
#               cost_le / num_batches, cost_ld / num_batches,)
#        predY = predict(model, val_dataset)
#        print("Epoch %d, cost = %f, acc = %.2f%%"
#              % (i + 1, cost / num_batches, 100. * np.mean(predY == teY)))

    torch.save(encoder.state_dict(), './encoder')
    torch.save(decoder.state_dict(), './decoder')

def train_wholenet(train_dataset, val_dataset, test_dataset):
    encoder_arch = [[6,14],[14,28],[28,28],[28,6]]
    decoder_arch = [[12,24],[24,28],[28,28],[28,7]]
    loss = nn.MSELoss()
    
    n_examples = len(train_dataset)  
    model = network.Net(encoder_arch, decoder_arch)
    
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
    
    batch_size = 500
    num_batches = n_examples // batch_size
    
    for i in range(5000):
        cost = 0.
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer, train_dataset[start: end])
        print("Epoch = {epoch}, cost = {le}".format(epoch = i+1, le = cost / num_batches))
        #predY = predict(model, val_dataset)
        #print("Epoch %d, cost = %f, acc = %.2f%%"
        #      % (i + 1, cost / num_batches, 100. * np.mean(predY == teY)))
    
    torch.save(model.state_dict(), './model')
    

def main():
    dataset = exoskeleton_dataset.ExoskeletonDataset(
            file="data/exoskeleton_data",root_dir="/")    
    train_dataset, val_dataset, test_dataset = dataset.GetDataset()
    
    train_separately(train_dataset, val_dataset, test_dataset)    
    print("Separate network train done! Begin whole network training...")
    
    train_wholenet(train_dataset, val_dataset, test_dataset)


if __name__ == "__main__":
    main()