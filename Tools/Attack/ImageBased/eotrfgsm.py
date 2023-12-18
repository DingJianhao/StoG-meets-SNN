import torch
import torch.nn as nn
# from torchattacks.attack import Attack
from ...Attack.base import BaseAttack

class EOTRFGSM(BaseAttack):
    def __init__(self, model, loss_function, forward_function=None, targeted=False, representation=None,
                 eps=8/255, alpha=4/255, repeat=10, **kwargs):
        super(EOTRFGSM, self).__init__(model, loss_function, forward_function, representation, targeted)
        self.eps = eps
        self.alpha = alpha
        self.kwargs = kwargs
        self.repeat = repeat

    def forward(self, imgs, labels):
        imgs = imgs.clone().detach().to(imgs)
        labels = labels.clone().detach().to(imgs)

        imgs = imgs + self.alpha*torch.randn_like(imgs).to(imgs)
        imgs = torch.clamp(imgs, min=0, max=1).detach()
        imgs.requires_grad = True
        rpst = self.data_representation(imgs)
        
        # out = self.model_forward(rpst)
        # loss = self.get_loss(out, labels)
        # grad = torch.autograd.grad(loss, imgs,
        #                            retain_graph=False, create_graph=False)[0]

        grad = 0
        for _ in range(self.repeat):
            out = self.model_forward(rpst) # model should be random model
            loss = self.get_loss(out, labels)
            grad += torch.autograd.grad(loss, imgs,
                                    retain_graph=False, create_graph=False)[0]
        grad /= self.repeat
        
        adv_images = imgs + (self.eps - self.alpha) * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        return adv_images