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
from utils import orthogonal_retraction, convex_constraint
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def val(model, test_loader, device, representation, T, atk=None, num_targets=10):
    correct = 0
    total = 0
    fail = 0
    original_right_total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        is_img = (len(inputs.shape) == 4)

        # clean predict for success rate
        if atk is not None:
            with torch.no_grad():
                if is_img:
                    rpst = representation(inputs)
                else:
                    if len(inputs.shape) == 5:
                        rpst = inputs
                    else:
                        rpst = representation(inputs) #inputs
                outputs = model(rpst).mean(0)
            _, predicted = outputs.max(1)
            mask = predicted.eq(targets).float()
        
        if atk is not None:
            atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            if atk.targeted:
                rd = torch.randint_like(targets, high=num_targets - 1)
                new_targets = (targets + rd) % num_targets

            if is_img:
                if atk.targeted:
                    inputs = atk(inputs, new_targets)
                else:
                    inputs = atk(inputs, targets)
            else:
                rpst = representation(inputs).clone().detach().to(inputs)
                if atk.targeted:
                    inputs = atk(rpst, new_targets)
                else:
                    inputs = atk(rpst, targets)
            atk.model.set_simulation_time(T)
        
        
        with torch.no_grad():
            if is_img:
                rpst = representation(inputs)
            else:
                if len(inputs.shape) == 5:
                    rpst = inputs
                else:
                    rpst = representation(inputs) #inputs
            outputs = model(rpst).mean(0)
        _, predicted = outputs.max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())

        if atk is not None:
            wrong = ~(predicted.eq(targets))
            original_right_total += mask.sum()
            fail += (wrong.float()*mask).sum()
    if atk is not None:
        success_rate = 100 * fail / original_right_total
    else:
        success_rate = 0
    final_acc = 100 * correct / total
    return final_acc, success_rate


def train(model, sparse_modules, device, train_loader, criterion, representation,
             sparse_prior, optimizer, sparse_optimizer, beta, gamma, T, atk=None, atk_mode='FGSM', sparse_step=True, parseval=False):
    running_loss = 0
    model.train()
    total = 0
    correct = 0
    tt = 0.
    # print(sparse_step)
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        targets = targets.to(device)
        inputs = inputs.to(device)

        if atk is not None:
            if atk_mode.lower() == 'fgsm':
                atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
                rpst = representation(inputs).clone().detach().to(inputs)
                rpst.requires_grad = True
                loss = criterion(model(rpst).mean(0), targets)
                grad = torch.autograd.grad(loss, rpst,
                                    retain_graph=False, create_graph=False)[0]
                adv_rpst = rpst + atk.eps * grad.sign()
                adv_rpst = torch.clamp(adv_rpst, min=0, max=1).detach()
                rpst = adv_rpst
            else:
                pass # mix
        else:
            rpst = representation(inputs)
        model.train()
        
        optimizer.zero_grad()
        if sparse_optimizer is not None and sparse_step:
            sparse_optimizer.zero_grad()

        outputs = model(rpst).mean(0)
        loss = criterion(outputs, targets)
        running_loss += loss.item()

        loss.mean().backward()
        optimizer.step()
        
        if sparse_optimizer is not None and sparse_step:
            sparse_loss = gamma * sparse_prior(sparse_modules)
            sparse_loss.backward()
            sparse_optimizer.step()
        
        if parseval:
            orthogonal_retraction(model, beta)
            convex_constraint(model)

        total += float(targets.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(targets.cpu()).sum().item())
    return running_loss, 100 * correct / total