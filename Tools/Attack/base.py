import torch.nn as nn


class BaseAttack(nn.Module):
    def __init__(self, model, loss_function, forward_function=None, representation=None, targeted=False, **kwargs):
        super(BaseAttack, self).__init__()
        self.model = model
        self.loss_function = loss_function
        self.forward_function = forward_function
        self.representation = representation
        self.targeted = targeted
        self.kwargs = kwargs

    def forward(self, events, labels):
        pass

    def model_forward(self, x):
        if self.forward_function is not None:
            outputs = self.forward_function(self.model, x, **self.kwargs)
        else:
            outputs = self.model(x)
        return outputs

    def data_representation(self, x):
        if self.representation is not None:
            return self.representation(x)
        else:
            return x

    def get_loss(self, outputs, labels):
        if self.targeted:
            cost = -self.loss_function(outputs, labels.long())
        else:
            cost = self.loss_function(outputs, labels.long())
        return cost

    def set_training_mode(self, model_training=False, batchnorm_training=False, dropout_training=False):
        if model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m.eval()
                if not dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m.eval()
        else:
            self.model.eval()