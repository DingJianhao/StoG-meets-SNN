import sys
sys.path.append('Tools')
import argparse
import os
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import data_loaders
import copy
from functions import seed_all, get_logger, BPTT_attack
import Tools.Attack as attack
from models import *
from models.layers import LIFSpike, SparsePoissonCoding
# from utils import train, val
from utils import sparse_prior_l1, sparse_prior_l2
from sparse_code_utils import val, train

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size') # 16 for gesture
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training. ')
parser.add_argument('--optim', default='sgd', type=str, help='model')
parser.add_argument('-suffix', '--suffix', default='', type=str, help='suffix')
parser.add_argument('-pretrain', '--pretrain', default='', type=str, help='pretrain') #vgg11_clean_l2[0.000500]

# model configuration
parser.add_argument('-data', '--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('-arch', '--model', default='vgg11', type=str, help='model')
parser.add_argument('-T', '--time', default=8, type=int, metavar='N', help='snn simulation time') # 10 for gesture
parser.add_argument('-tau', '--tau', default=1.0, type=float, help='tau')
parser.add_argument('-allow', '--allow_sparse', default=1, type=int, metavar='N', help='allow sparse')

# training configuration
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-lr', '--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-dev', '--device', default='0', type=str, help='device')

# adv training configuration
parser.add_argument('-special', '--special', default='l2', type=str, help='[l2, parseval]')
parser.add_argument('-beta', '--beta', default=5e-4, type=float, help='regulation beta')
parser.add_argument('-atk', '--attack', default='fgsm', type=str, help='attack')
parser.add_argument('-eps', '--eps', default=2, type=float, metavar='N', help='attack eps') #
parser.add_argument('-atk_m', '--attack_mode', default='', type=str, help='attack mode (_, FGSM, HIRE)')

# sparisity
parser.add_argument('-K', '--K', default=100, type=float, metavar='N', help='epoch')
parser.add_argument('-gamma', '--gamma', default=5e-6, type=float, help='float')
parser.add_argument('-prior', '--prior', default='l2', type=str, help='prior')

# only PGD
# parser.add_argument('-alpha', '--alpha', default=1, type=float, metavar='N', help='pgd attack alpha')
# parser.add_argument('-steps', '--steps', default=2, type=int, metavar='N', help='pgd attack steps')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VoxelGridMean(VoxelGrid):
    def __init__(self, T, H, W, downsample=1, sum=False, start_time=None, end_time=None, temporal_dim_first=False, scale=None):
        super(VoxelGridMean, self).__init__(T, H, W, downsample, sum, start_time, end_time, temporal_dim_first, scale)
    
    def forward(self, events, enable_grad=True, **kwargs):
        x = super().forward(events, enable_grad, **kwargs) # b, t, c, h, w
        x = x.mean(1).unsqueeze_(1).repeat(1, self.T, 1, 1, 1)
        x[:,:,1,:,:] = x[:,:,0,:,:]
        return x

def main():
    global args
    if args.dataset.lower() == 'cifar10':
        use_cifar10 = True
        num_labels = 10
        init_c = 3
        H=32
    elif args.dataset.lower() == 'cifar100':
        use_cifar10 = False
        num_labels = 100
        init_c = 3
        H=32
    elif args.dataset.lower() == 'svhn':
        num_labels = 10
        init_c = 3
        H=28
    elif args.dataset.lower() == 'tinyimagenet':
        num_labels = 200
        init_c = 3
        H=64

    # >>>>>>>IMPORTANT<<<<<<<< Edit log_dir
    log_dir = '%s-checkpoints' % (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed_all(args.seed)
    if 'cifar' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_cifar(use_cifar10=use_cifar10)
    if 'tinyimagenet' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_tinyimagenet()
    if args.dataset.lower() == 'svhn':
        train_dataset, val_dataset, znorm = data_loaders.build_svhn()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers, pin_memory=True)
    
        

    if 'vgg' in args.model.lower():
        model = VGG(args.model.lower(), args.time, num_labels, znorm, init_c=init_c, tau=args.tau, H=H)
    elif 'wideresnet' in args.model.lower():
        model = WideResNet(args.model.lower(), args.time, num_labels, znorm, init_c=init_c, tau=args.tau)
    elif 'resnet' in args.model.lower() and 'wide' not in args.model.lower(): # for rebuttal
        model = ResNet17(args.time, num_labels, znorm, tau=args.tau)
    else:
        raise AssertionError("model not supported")

    model.set_simulation_time(args.time)
    # state_dict = torch.load(os.path.join(log_dir, args.pretrain + '.pth'), map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
    model.to(device)

    def replace(model):
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                model._modules[name] = replace(module)
                if module.__class__.__name__ == 'LIFSpike':
                    model._modules[name] = nn.Sequential(
                        LIFSpike(T = module.T, thresh = module.thresh, tau = module.tau, gama = module.gama),
                        SparsePoissonCoding(T=module.T)
                    )
        return model
    
    def get_mean_p(modules):
        x = 0
        for m in modules:
            x += m.htanh(m.poisson_rate.data).mean().item()
        return x / len(modules)

    if args.allow_sparse:
        model = replace(model)

    sparse_modules = nn.ModuleList()
    non_sparse_modules = nn.ModuleList()
    for m in model.modules():
        if isinstance(m, SparsePoissonCoding):
            sparse_modules.append(m)
        elif isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d)):
            non_sparse_modules.append(m)

    # initialize new module
    model.eval()
    if 'cifar' in args.dataset.lower():
        model(torch.rand(5, 3, 32, 32).to(device))
    elif 'tinyimagenet' in args.dataset.lower():
        model(torch.rand(5, 3, 64, 64).to(device))
    else:
        model(torch.rand(16, 10, 2, 128, 128).to(device))
    model.to(device)

    ff = BPTT_attack # default
    if 'cifar' in args.dataset.lower() or 'tinyimagenet' in args.dataset.lower():
        rpst = lambda x: x
    else:
        rpst = represent_module
    
    if len(args.attack_mode) > 0:
        atk = attack.FGSM(model, forward_function = ff, eps = args.eps / 255,
            loss_function = nn.CrossEntropyLoss(), representation = rpst, T = args.time)
    else:
        atk = None


    criterion = nn.CrossEntropyLoss().to(device)

    if args.optim.lower() == 'adam' and args.special == 'l2':
        optimizer = torch.optim.Adam(non_sparse_modules.parameters(), lr=args.lr, weight_decay=args.beta)
        if args.allow_sparse:
            sparse_optimizer = torch.optim.Adam(sparse_modules.parameters(), lr=args.lr, weight_decay=0)
    elif args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(non_sparse_modules.parameters(), lr=args.lr)
        if args.allow_sparse:
            sparse_optimizer = torch.optim.Adam(sparse_modules.parameters(), lr=args.lr, weight_decay=0)
    elif args.optim.lower() == 'sgd' and args.special == 'l2':
        optimizer = torch.optim.SGD(non_sparse_modules.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.beta)
        if args.allow_sparse:
            sparse_optimizer = torch.optim.SGD(sparse_modules.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    elif args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(non_sparse_modules.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
        if args.allow_sparse:
            sparse_optimizer = torch.optim.SGD(sparse_modules.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    enable_parseval = (args.special.lower() == 'parseval')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.allow_sparse:
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(sparse_optimizer, T_max=args.epochs)
    best_acc = 0

    if not args.allow_sparse:
        sparse_optimizer = None

    # IMPORTANT<<<<<<<<<<<<< modifed
    identifier = args.model
    if args.tau != 1.0:
        identifier += '_tau%.2f' % args.tau
    if atk is not None:
        identifier += '_%s' % (atk.__class__.__name__)
    else:
        identifier += '_clean'

    if args.allow_sparse:
        identifier += '_s'

    identifier += '_%s[%f]' % (args.special, args.beta)
    if len(args.prior) > 0 and args.allow_sparse:
        identifier += '_s%s[%d,%f]' % (args.prior, args.K, args.gamma)
    identifier += args.suffix

    # parseval = (args.special == 'parseval')

    logger = get_logger(os.path.join(log_dir, '%s.log' % (identifier)))
    logger.info('start training!')

    print(identifier)
    if args.prior == 'l2':
        sparse_prior = sparse_prior_l2
    else:
        sparse_prior = sparse_prior_l1

    for epoch in range(args.epochs):
        if epoch < args.K:
            sparse_gamma = args.gamma
            sparse_step = True
        else:
            sparse_gamma = 0 # not updating p
            sparse_step = False
        loss, acc = train(model, sparse_modules, device, train_loader, criterion, rpst,
                             sparse_prior, optimizer, sparse_optimizer, beta=args.beta, gamma=sparse_gamma, T=args.time, 
                             atk=atk, atk_mode=args.attack_mode, sparse_step=sparse_step, parseval=enable_parseval)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, args.epochs, loss, acc))
        scheduler.step()
        if args.allow_sparse:
            scheduler2.step()
        tmp, _ = val(model, test_loader, device, rpst, args.time)
        if args.allow_sparse:
            mean_p = get_mean_p(sparse_modules)
            logger.info('Epoch:[{}/{}]\t Test acc={:.3f} mean_p:{:.5f}'.format(epoch, args.epochs, tmp, mean_p))
        else:
            logger.info('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch, args.epochs, tmp))

        if best_acc < tmp:
            best_acc = tmp
            torch.save(model.state_dict(), os.path.join(log_dir, '%s.pth' % (identifier)))

    logger.info('Best Test acc={:.3f}'.format(best_acc))


if __name__ == "__main__":
    main()
