import torch
import torch.nn as nn
# from torchattacks.attack import Attack
from ...Attack.base import BaseAttack

class RFGSM(BaseAttack):
    def __init__(self, model, loss_function, forward_function=None, targeted=False, representation=None,
                 eps=8/255, alpha=4/255, **kwargs):
        super(RFGSM, self).__init__(model, loss_function, forward_function, representation, targeted)
        self.eps = eps
        self.alpha = alpha
        self.kwargs = kwargs

    def forward(self, imgs, labels):
        imgs = imgs.clone().detach().to(imgs)
        labels = labels.clone().detach().to(imgs)

        imgs = imgs + self.alpha*torch.randn_like(imgs).to(imgs)
        imgs = torch.clamp(imgs, min=0, max=1).detach()
        imgs.requires_grad = True
        rpst = self.data_representation(imgs)
        out = self.model_forward(rpst)
        loss = self.get_loss(out, labels)
        grad = torch.autograd.grad(loss, imgs,
                                   retain_graph=False, create_graph=False)[0]
        adv_images = imgs + (self.eps - self.alpha) * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        return adv_images

# class RFGSM(Attack):
#     r"""
#     altered from torchattack
#     """
#     def __init__(self, model, forward_function=None, eps=8/255, T=None, **kwargs):
#         super().__init__("RFGSM", model)
#         self.eps = eps
#         self.alpha = eps/2
#         self._supported_mode = ['default', 'targeted']
#         self.forward_function = forward_function
#         self.T = T

#     def forward(self, images, labels):
#         r"""
#         Overridden.
#         """
#         images = images.clone().detach().to(self.device)
#         labels = labels.clone().detach().to(self.device)

#         if self._targeted:
#             target_labels = self._get_target_label(images, labels)

#         loss = nn.CrossEntropyLoss()

#         ### modified
#         adv_images = images + self.alpha*torch.randn_like(images)
#         adv_images = torch.clamp(adv_images, min=0, max=1).detach()

#         adv_images.requires_grad = True
#         if self.forward_function is not None:
#             outputs = self.forward_function(self.model, adv_images, self.T)
#         else:
#             outputs = self.model(adv_images)

#         # Calculate loss
#         if self._targeted:
#             cost = -loss(outputs, target_labels)
#         else:
#             cost = loss(outputs, labels)

#         # Update adversarial images
#         grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
#         adv_images = torch.clamp(adv_images.detach() + (self.eps - self.alpha)*grad.sign(), min=0, max=1).detach()

#         return adv_images
