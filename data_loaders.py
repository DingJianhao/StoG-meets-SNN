import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST, SVHN
import warnings
import os
from torch.utils.data import random_split
from os import listdir
import numpy as np
from os.path import isfile, join
import json
import sys
sys.path.append('../NeuromorphicIntelligenceTools_main')
from NeuromorphicIntelligenceTools.Dataset.NMVision.DVSGesture import DVSGesture

from spikingjellylocal.datasets.dvs128_gesture import DVS128Gesture

warnings.filterwarnings('ignore')

# code from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py 
# Improved Regularization of Convolutional Neural Networks with Cutout.
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def build_cifar(cutout=True, use_cifar10=True, download=True):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()]

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        # aug.append(
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = CIFAR10(root='~/dataset/cifar10',
                                train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root='~/dataset/cifar10',
                              train=False, download=download, transform=transform_test)
        norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        # aug.append(
        #     transforms.Normalize(
        #         (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        # )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root='~/dataset/cifar100',
                                 train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root='~/dataset/cifar100',
                               train=False, download=download, transform=transform_test)
        norm = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    return train_dataset, val_dataset, norm

def build_dvsgesture(T=10):
    DATA_DIR = '/home/dingjh/dataset/DVS Gesture dataset/'
    def t(data):
        aug = transforms.Compose([transforms.Resize(size=(64, 64)), transforms.RandomHorizontalFlip()])
        data = torch.from_numpy(data)
        data = aug(data).float()
        return data
    
    def tt(data):
        aug = transforms.Resize(size=(64, 64))
        data = torch.from_numpy(data)
        data = aug(data).float()
        return data

    train_dataset = DVS128Gesture(root=DATA_DIR, train=True, data_type='frame', frames_number=T, split_by='number', transform=t)
    val_dataset = DVS128Gesture(root=DATA_DIR, train=False, data_type='frame', frames_number=T, split_by='number', transform=tt)
    norm = 'jelly'
    return train_dataset, val_dataset, norm

def build_gesture(T=10):
    DATA_DIR = '~/dataset/DVS Gesture dataset/events_np'
    train_dataset = DVSGesture(root=DATA_DIR, train=True)

    val_dataset = DVSGesture(root=DATA_DIR, train=False)
    # train_data_loader = NeuromorphicVisionLoader(dataset=train_dataset,
    #                                             shuffle=True,
    #                                             batch_size=batch_size,
    #                                             num_workers=num_workers, )
    # test_data_loader = NeuromorphicVisionLoader(dataset=test_dataset,
    #                                             shuffle=False,
    #                                             batch_size=batch_size,
    #                                             num_workers=num_workers)
    norm = ((0.0, 0.0), (1.0, 1.0))
    return train_dataset, val_dataset, norm

def build_svhn(cutout=True, download=False):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(cutout(n_holes=1, length=16))

    transform_train = transforms.Compose(aug)
    transform_test = transforms.ToTensor()
    train_dataset = SVHN(root='G:/Dataset/svhn',
                            split='train', download=download, transform=transform_train)
    val_dataset = SVHN(root='G:/Dataset/svhn',
                          split='test', download=download, transform=transform_test)

    # dataset = SVHN(root=dataset_dir['SVHN'],
    #     split='train', download=download, transform=transform_train)
    # random_seed = 42
    # torch.manual_seed(random_seed)
    # val_size = 5000
    # train_size = len(dataset) - val_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    norm = ((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    return train_dataset, val_dataset, norm


# def build_mnist(download=False):
#     train_dataset = MNIST(root=dataset_dir['MNIST'],
#                              train=True, download=download, transform=transforms.ToTensor())
#     val_dataset = MNIST(root=dataset_dir['MNIST'],
#                            train=False, download=download, transform=transforms.ToTensor())
#     return train_dataset, val_dataset

# def GetDVSCifar10():
#     root = '/home/butong/datasets/CIFAR10DVS/'
#     data1 = CIFAR10DVS(root=root, data_type='frame', frames_number=10, split_by='number', transform=trans_t)
#     train_dataset, _ = torch.utils.data.random_split(data1, [9000, 1000], generator=torch.Generator().manual_seed(42))
#     data2 = CIFAR10DVS(root=root, data_type='frame', frames_number=10, split_by='number', transform=trans)
#     _, test_dataset = torch.utils.data.random_split(data2, [9000, 1000], generator=torch.Generator().manual_seed(42))
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
#     return train_loader, test_loader

# def build_imagenet():
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     root = '/data_smr/dataset/ImageNet'
#     train_root = os.path.join(root, 'train')
#     val_root = os.path.join(root, 'val')
#     train_dataset = ImageFolder(
#         train_root,
#         transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ])
#     )
#     val_dataset = ImageFolder(
#         val_root,
#         transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             normalize,
#         ])
#     )
#     return train_dataset, val_dataset

class TinyImagenet(torchvision.datasets.ImageFolder):
    def __init__(self, root: str, train=True, transform=None):
        import os
        from collections import defaultdict
        if train:
            super().__init__(root=os.path.join(root, 'train'), transform=transform)
        else:
            super().__init__(root=os.path.join(root, 'val'), transform=transform)
            val_set = defaultdict(list)
            with open(os.path.join(root, 'val', 'val_annotations.txt'), 'r') as txt:
                for line in txt:
                        file = line.strip('\n').split('\t')[0]
                        label = line.strip('\n').split('\t')[1]
                        val_set[label].append(file)
            self.classes = sorted(val_set.keys())
            self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
            self.samples = []
            for c in self.classes:
                for file in val_set[c]:
                    self.samples.append((os.path.join(root, 'val', 'images', file), self.class_to_idx[c]))
            self.targets = [s[1] for s in self.samples]

def build_tinyimagenet(cutout=True, download=False):
    aug = [transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()]

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))
    transform_train = transforms.Compose(aug)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = TinyImagenet(root='/home/dingjh/dataset/tiny-imagenet-200',
                            train=True, transform=transform_train)
    val_dataset = TinyImagenet(root='/home/dingjh/dataset/tiny-imagenet-200',
                            train=False, transform=transform_test)
    norm = ((0.480, 0.448, 0.398), (0.272, 0.266, 0.274))
    return train_dataset, val_dataset, norm

if __name__ == '__main__':
    # train_set, test_set = build_mnist(download=True)
    # train_set, test_set, norm = build_svhn(download=True)
    x = build_dvsgesture(T=8)
    print(x[0][0][0].max())