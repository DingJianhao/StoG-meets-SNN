import torch
from ...Attack.base import BaseAttack

class PGD(BaseAttack):
    def __init__(self, model, loss_function, forward_function=None, targeted=False, representation=None,
                 eps=0.3, alpha=2/255, steps=40, random_start=True, **kwargs):
        super(PGD, self).__init__(model, loss_function, forward_function, representation, targeted)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.kwargs = kwargs
        self.random_start = random_start

    def forward(self, imgs, labels):
        imgs = imgs.clone().detach().to(imgs)
        labels = labels.clone().detach().to(imgs)
        adv_imgs = imgs.clone().detach()
        if self.random_start:
            adv_imgs = adv_imgs + torch.empty_like(adv_imgs).uniform_(-self.eps, self.eps)
            adv_imgs = torch.clamp(adv_imgs, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_imgs.requires_grad = True
            rpst = self.data_representation(adv_imgs)
            out = self.model_forward(rpst)
            loss = self.get_loss(out, labels)
            grad = torch.autograd.grad(loss, adv_imgs,
                                       retain_graph=False, create_graph=False)[0]
            adv_imgs = adv_imgs.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_imgs - imgs, min=-self.eps, max=self.eps)
            adv_imgs = torch.clamp(imgs + delta, min=0, max=1).detach()
        return adv_imgs

