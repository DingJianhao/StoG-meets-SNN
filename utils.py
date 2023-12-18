# from cv2 import mean
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from models.layers import *
import random
import time
from torchvision.utils import make_grid
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



def train(model, device, train_loader, criterion, optimizer, T, atk, beta, parseval=False):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    tt = 0.
    for i, (images, labels) in enumerate((train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        st = time.time()
        if atk is not None:
            atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            images = atk(images, labels)
        tt += (time.time()-st)
        if T > 0:
            outputs = model(images).mean(0)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        if parseval:
            orthogonal_retraction(model, beta)
            convex_constraint(model)
    
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    print(tt)
    return running_loss, 100 * correct / total

def train_mix(model, device, train_loader, criterion, optimizer, T, atk_list, beta, parseval=False):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    tt = 0.
    for i, (images, labels) in enumerate((train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        # st = time.time()
        if atk_list is not None:
            atk = atk_list[random.randint(0, len(atk_list)-1)]
            atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            images = atk(images, labels)
        # tt += time.time() - st
        if T > 0:
            outputs = model(images).mean(0)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        if parseval:
            orthogonal_retraction(model, beta)
            convex_constraint(model)

        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    # print(tt)
    return running_loss, 100 * correct / total

def val(model, test_loader, device, T, atk=None, representation=None):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate((test_loader)):
        inputs = inputs.to(device)

        if atk is not None:
            atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            inputs = atk(inputs, targets.to(device))
            model.set_simulation_time(T)
        with torch.no_grad():
            if T > 0:
                outputs = model(inputs).mean(0)
            else:
                outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
    final_acc = 100 * correct / total
    return final_acc


def sparse_prior_l2(model_list):
        x = 0
        for m in model_list.modules():
            if isinstance(m, SparsePoissonCoding):
                x += torch.sum(torch.pow(m.poisson_rate, 2)) * 0.5
        return x

def sparse_prior_l1(model_list):
        x = 0
        for m in model_list.modules():
            if isinstance(m, SparsePoissonCoding):
                x += torch.sum(torch.abs(m.poisson_rate))
        return x

def orthogonal_retraction(model, beta=0.002):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if isinstance(module, nn.Conv2d):
                    weight_ = module.weight.data
                    sz = weight_.shape
                    weight_ = weight_.reshape(sz[0],-1)
                    rows = list(range(module.weight.data.shape[0]))
                elif isinstance(module, nn.Linear):
                    if module.weight.data.shape[0] < 200: # set a sample threshold for row number
                        weight_ = module.weight.data
                        sz = weight_.shape
                        weight_ = weight_.reshape(sz[0], -1)
                        rows = list(range(module.weight.data.shape[0]))
                    else:
                        rand_rows = np.random.permutation(module.weight.data.shape[0])
                        rows = rand_rows[: int(module.weight.data.shape[0] * 0.3)]
                        weight_ = module.weight.data[rows,:]
                        sz = weight_.shape
                module.weight.data[rows,:] = ((1 + beta) * weight_ - beta * weight_.matmul(weight_.t()).matmul(weight_)).reshape(sz)


def convex_constraint(model):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, ConvexCombination):
                comb = module.comb.data
                alpha = torch.sort(comb, descending=True)[0]
                k = 1
                for j in range(1,module.n+1):
                    if (1 + j * alpha[j-1]) > torch.sum(alpha[:j]):
                        k = j
                    else:
                        break
                gamma = (torch.sum(alpha[:k]) - 1)/k
                module.comb.data -= gamma
                torch.relu_(module.comb.data)


def val_success_rate(model, test_loader, device, atk=None):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate((test_loader)):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        mask = predicted.eq(targets).float()
        
        if atk is not None:
            atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            inputs = atk(inputs, targets.to(device))
            
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        
        predicted = ~(predicted.eq(targets))
        total += mask.sum()
        correct += (predicted.float()*mask).sum()

    final_acc = 100 * correct / total
    return final_acc.item()

