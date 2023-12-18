import torch
from ...Attack.base import BaseAttack

class FGSM(BaseAttack):
    def __init__(self, model, loss_function, forward_function=None, targeted=False, representation=None,
                 eps=0.007, **kwargs):
        super(FGSM, self).__init__(model, loss_function, forward_function, representation, targeted)
        self.eps = eps
        self.kwargs = kwargs

    def forward(self, imgs, labels):
        imgs = imgs.clone().detach().to(imgs)
        labels = labels.clone().detach().to(imgs)
        imgs.requires_grad = True
        rpst = self.data_representation(imgs)
        out = self.model_forward(rpst)
        # print(out)
        loss = self.get_loss(out, labels)
        # print(loss)
        grad = torch.autograd.grad(loss, imgs,
                                   retain_graph=False, create_graph=False)[0]
        # print(grad.max())
        adv_images = imgs + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        return adv_images