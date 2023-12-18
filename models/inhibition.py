import torch
import torch.nn as nn
from models.utils import *
# from models.layers import ExpandTemporalDim, MergeTemporalDim

class PoissonCoding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ndim):
        ctx.save_for_backward(input)
        shape = input.shape
        rand = torch.cuda.FloatTensor(ndim, *shape).uniform_()
        return (rand < input.unsqueeze(0)).float()

    @staticmethod
    def backward(ctx, grad_output):
        # (input) = ctx.saved_tensors
        # print(grad_output.shape)
        # print(grad_output.mean(dim=0).shape)
        return grad_output.mean(dim=0), None

class PoissonCodingPerSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ndim):
        # print(input.shape, ndim)
        ctx.save_for_backward(input)
        shape = input.shape # N, C, H, W
        rand = torch.rand(ndim,*shape).to(input)
        return (rand < input.unsqueeze(0)).float()

    @staticmethod
    def backward(ctx, grad_output):
        # (input) = ctx.saved_tensors
        # print(grad_output.shape)
        # print(grad_output.mean(dim=0).shape)
        return grad_output.mean(dim=0), None

# class SparsePoissonCoding(nn.Module):
#     def __init__(self, T, size=None):
#         super(SparsePoissonCoding, self).__init__()
#         self.T = T
#         if size is None:
#             self.poisson_rate = None
#         else: # eg size=(3, 32, 32)
#             self.poisson_rate = torch.nn.Parameter(torch.ones(size))
#         self.attack_mode = False
#
#     def forward(self, x):
#         if not self.attack_mode:
#             if self.poisson_rate is None: # first time initialize
#                 self.poisson_rate = torch.nn.Parameter(torch.ones(x.shape[1:]))
#                 self.to(x)
#             mask = PoissonCoding.apply(self.poisson_rate, x.shape[0])
#             # print(mask.shape, x.shape)
#             return x * mask
#         else:
#             return x
#
#     def reset(self):
#         self.poisson_rate.data = torch.ones_like(self.poisson_rate).to(self.poisson_rate)
#
#     def set_atk_mode(self, mode=False):
#         self.attack_mode = mode

class inhibition_module(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rate, ndim, temperature):
        eps = 1e-3
        shape = rate.shape
        uniform = torch.cuda.FloatTensor(ndim, *shape).uniform_()
        alpha = rate / (1 - rate + eps)
        expL = uniform / (1 - uniform + eps)
        sigma = 1 / (1 + torch.exp( -(torch.log(alpha.unsqueeze(0)) + torch.log(expL)) / temperature))
        ctx.save_for_backward(sigma, rate, torch.tensor(temperature))
        return (-torch.log(expL) <= torch.log(alpha.unsqueeze(0))).float()

    @staticmethod
    def backward(ctx, grad_output):
        (sigma, inhibition_rate, temperature) = ctx.saved_tensors
        grad_in = (grad_output * sigma * (1 - sigma)).mean(dim=0) * (-1 / (temperature * inhibition_rate *
                                                   (1 - inhibition_rate)))
        # print(grad_in.shape)
        return grad_in, None, None


class SparsePoissonCoding(nn.Module): # version 2
    def __init__(self, T, size=None):
        # The Concrete Distribution:
        # A Continuous Relaxation of
        # Discrete Random Variables
        super(SparsePoissonCoding, self).__init__()
        self.T = T
        if size is None:
            self.poisson_rate = None
        else:
            self.poisson_rate = torch.nn.Parameter(torch.ones(size))
        self.attack_mode = False
        self.htanh = nn.Hardtanh(min_val=1e-1, max_val=1)

    def forward(self, x):
        if not self.attack_mode:
            if self.poisson_rate is None: # first time initialize
                self.poisson_rate = torch.nn.Parameter(torch.ones(x.shape[1:]))
                self.to(x)
            inhibition = inhibition_module.apply(self.htanh(self.poisson_rate), x.shape[0], 0.5)
            return x * inhibition
            # return x * self.htanh(self.poisson_rate).unsqueeze(0)
        else:
            return x

    def reset(self):
        self.poisson_rate.data = torch.ones_like(self.poisson_rate).to(self.poisson_rate)

    def set_atk_mode(self, mode=False):
        self.attack_mode = mode


class SparsePoissonCodingPerSample(nn.Module):
    def __init__(self, T, imgs, size=(3, 32, 32)):
        super(SparsePoissonCodingPerSample, self).__init__()
        self.T = T
        self.size = size
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        N = imgs.shape[0]
        self.poisson_rate = torch.nn.Parameter(torch.ones((N, *self.size)))

    def forward(self, x):
        x = self.expand(x)
        # print(x.shape, self.poisson_rate.shape)
        mask = PoissonCodingPerSample.apply(self.poisson_rate, x.shape[0])
        # print(mask.shape)
        x_ = x * mask
        x_ = self.merge(x_)
        return x_

if __name__ == '__main__':
    m = torch.distributions.RelaxedBernoulli(temperature=0.005, probs=torch.tensor([0.5, 0.7])).sample()
    print(m)