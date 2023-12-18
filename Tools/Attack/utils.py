import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron
from copy import deepcopy


def enable_eval_grad(model): # run before spikingjelly attack
    model.eval()
    for module in model.modules():
        if isinstance(module, neuron.BaseNode):
            module.train()
