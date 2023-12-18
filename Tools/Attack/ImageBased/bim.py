import torch
from ...Attack.base import BaseAttack


class BIM(BaseAttack):
    def __init__(self, model, loss_function, forward_function=None, targeted=False, representation=None,
                 eps=0.3, alpha=2/255, steps=40, **kwargs):
        super(BIM, self).__init__(model, loss_function, forward_function, representation, targeted)
        self.eps = eps
        self.alpha = alpha
        self.kwargs = kwargs
        if steps == 0:
            self.steps = int(min(eps * 255 + 4, 1.25 * eps * 255))
        else:
            self.steps = steps

    def forward(self, imgs, labels):
        imgs = imgs.clone().detach().to(imgs)
        labels = labels.clone().detach().to(imgs)
        ori_imgs = imgs.clone().detach().to(imgs)

        for _ in range(self.steps):
            imgs.requires_grad = True
            rpst = self.data_representation(imgs)
            out = self.model_forward(rpst)
            loss = self.get_loss(out, labels)
            grad = torch.autograd.grad(loss, imgs,
                                       retain_graph=False, create_graph=False)[0]
            adv_images = imgs + self.alpha * grad.sign()
            a = torch.clamp(adv_images - self.eps, min=0)
            b = (adv_images >= a).float() * adv_images \
                + (adv_images < a).float() * a
            c = (b > ori_imgs + self.eps).float() * (ori_imgs + self.eps) \
                + (b <= ori_imgs + self.eps).float() * b
            imgs = torch.clamp(c, max=1).detach()
        return imgs
