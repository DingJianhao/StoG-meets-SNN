# from sklearn.semi_supervised import SelfTrainingClassifier
from models.layers import *

cfg = {
    'vgg5' : [[64, 'A'], 
              [128, 128, 'A'],
              [],
              [],
              []],
    'vgg11': [
        [64, 'A'],
        [128, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512],
        []
    ],
    'vgg13': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 'A'],
        [512, 512, 'A'],
        [512, 512, 'A']
    ],
    'vgg16': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512, 512, 'A']
    ],
    'vgg19': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 256, 256, 'A'],
        [512, 512, 512, 512, 'A'],
        [512, 512, 512, 512, 'A']
    ]
}

class VGG(nn.Module):
    def __init__(self, vgg_name, T, num_class, norm, init_c=3, tau=1.0, H=32):
        super(VGG, self).__init__()
        if norm is not None and isinstance(norm, tuple):
            self.norm = TensorNormalization(*norm)
        else:
            self.norm = TensorNormalization((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.T = T
        self.init_channels = init_c
        self.tau = tau
        self.jelly = (norm == 'jelly')

        if vgg_name == 'vgg11' or vgg_name == 'vgg5':
            self.W = 16
        else:
            self.W = 1
        
        if init_c == 2 and H==128: # event data
            self.rescale = nn.AdaptiveAvgPool2d((48, 48))
            self.W = 36

        if init_c == 1 and H==128: # event data
            self.rescale = nn.AdaptiveAvgPool2d((48, 48))
            self.W = 36
        
        self.layer1 = self._make_layers(cfg[vgg_name][0])
        self.layer2 = self._make_layers(cfg[vgg_name][1])
        self.layer3 = self._make_layers(cfg[vgg_name][2])
        self.layer4 = self._make_layers(cfg[vgg_name][3])
        self.layer5 = self._make_layers(cfg[vgg_name][4])
        self.classifier = self._make_classifier(num_class)
        
        if H == 64:
            self.layer1[0].stride = 2

        self.merge = MergeTemporalDim(T)
        self.expand = ExpandTemporalDim(T)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
        

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'A':
                layers.append(nn.AvgPool2d(2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(LIFSpike(self.T, tau=self.tau))
                self.init_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self, num_class):
        layer = [nn.Flatten(), nn.Linear(512*self.W, 4096), LIFSpike(self.T, tau=self.tau), nn.Linear(4096, 4096), LIFSpike(self.T, tau=self.tau), nn.Linear(4096, num_class)]    
        return nn.Sequential(*layer)
    
    # pass T to determine whether it is an ANN or SNN
    def set_simulation_time(self, mode='bptt', *args, **kwargs):
        # self.T = T
        for module in self.modules():
            if isinstance(module, (LIFSpike, ExpandTemporalDim)):
                # module.T = T
                if isinstance(module, LIFSpike):
                    module.mode = mode
        return

    def forward(self, input):
        if self.jelly:
            input = self.merge(input)
            out = self.layer1(input)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = torch.flatten(out, 1)
            out = self.classifier(out)
            out = self.expand(out)
            return out
        else:
            if len(input.shape) == 4:
                input = self.norm(input)
                if self.T > 0:
                    input = add_dimention(input, self.T)
                    input = self.merge(input)
            else:
                # print(input.shape)
                input = input.permute((1, 0, 2, 3, 4))
                input = self.merge(input)
                input = self.rescale(input)
            out = self.layer1(input)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = self.classifier(out)
            if self.T > 0:
                out = self.expand(out)
            return out
    

# class VGG_woBN(nn.Module):
#     def __init__(self, vgg_name, T, num_class, norm, dropout=0.1, init_c=3):
#         super(VGG_woBN, self).__init__()
#         if norm is not None and isinstance(norm, tuple):
#             self.norm = TensorNormalization(*norm)
#         else:
#             self.norm = TensorNormalization((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         self.T = T
#         self.init_channels = init_c
#         self.dropout = dropout
#
#         if "wobn" in vgg_name:
#             vgg_name = 'vgg5'
#         if vgg_name == 'vgg11' or vgg_name == 'vgg5':
#             self.W = 16
#         else:
#             self.W = 1
#
#         self.layer1 = self._make_layers(cfg[vgg_name][0])
#         self.layer2 = self._make_layers(cfg[vgg_name][1])
#         self.layer3 = self._make_layers(cfg[vgg_name][2])
#         self.layer4 = self._make_layers(cfg[vgg_name][3])
#         self.layer5 = self._make_layers(cfg[vgg_name][4])
#         self.classifier = self._make_classifier(num_class)
#
#         self.merge = MergeTemporalDim(T)
#         self.expand = ExpandTemporalDim(T)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.Linear):
#                 nn.init.zeros_(m.bias)
#
#     def _make_layers(self, cfg):
#         layers = []
#         for x in cfg:
#             if x == 'A':
#                 layers.append(nn.AvgPool2d(2))
#             else:
#                 layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
#                 layers.append(LIFSpike(self.T))
#                 layers.append(nn.Dropout(self.dropout))
#                 self.init_channels = x
#         return nn.Sequential(*layers)
#
#     def _make_classifier(self, num_class):
#         layer = [nn.Flatten(), nn.Linear(512*self.W, 4096), LIFSpike(self.T), nn.Linear(4096, 4096), LIFSpike(self.T), nn.Linear(4096, num_class)]
#         return nn.Sequential(*layer)
#
#     # pass T to determine whether it is an ANN or SNN
#     def set_simulation_time(self, T, mode='bptt'):
#         self.T = T
#         for module in self.modules():
#             if isinstance(module, (LIFSpike, ExpandTemporalDim)):
#                 module.T = T
#                 if isinstance(module, LIFSpike):
#                     module.mode = mode
#         return
#
#     def forward(self, input):
#         input = self.norm(input)
#         if self.T > 0:
#             input = add_dimention(input, self.T)
#             input = self.merge(input)
#         out = self.layer1(input)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         out = self.classifier(out)
#         if self.T > 0:
#             out = self.expand(out)
#         return out

if __name__ == '__main__':
    n = VGG('vgg11', T=10, num_class=11, norm=((0.4914, 0.4822), (0.2023, 0.1994)), init_c=2)
    print(n(torch.rand(5, 10, 2, 128, 128)).shape)