import sys
sys.path.append('Tools')
import argparse
import os

from models.VGG import *
import data_loaders
from functions import seed_all, BPTT_attack, get_logger
from sparse_code_utils import *
from models import *
import Tools.Attack as attack
import copy
import torch
import json

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
# just use default setting
parser.add_argument('-j','--workers',default=4, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=64, type=int,metavar='N',help='mini-batch size')
parser.add_argument('-sd', '--seed',default=42,type=int,help='seed for initializing training.')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')
parser.add_argument('-seed_in_dir', '--seed_in_dir', default=0, type=int, metavar='N', help='seed display in log dir')

# model configuration
parser.add_argument('-data', '--dataset', default='cifar10',type=str,help='dataset')
parser.add_argument('-arch','--model', default='vgg11', type=str,help='model')
parser.add_argument('-T','--time', default=8, type=int, metavar='N',help='snn simulation time')
parser.add_argument('-id', '--identifier', default='finetune-vgg11_FGSM[0.007843]_l2[0.000500]', type=str, help='model statedict identifier')
parser.add_argument('-config', '--config', default='standard', type=str,help='test configuration file')
parser.add_argument('-allow', '--allow_sparse', default=1, type=int, metavar='N', help='allow sparse')

# training configuration
parser.add_argument('-dev','--device',default='0',type=str,help='device')

# adv atk configuration
parser.add_argument('-atk','--attack',default='fgsm', type=str,help='attack')
parser.add_argument('-eps','--eps',default=8,type=float,metavar='N', help='attack eps')
parser.add_argument('-atk_m','--attack_mode',default='', type=str, help='attack mode')

parser.add_argument('-target','--target', default=0, type=float, metavar='N', help='target attack')

# only pgd
parser.add_argument('-alpha','--alpha',default=2.55/1,type=float,metavar='N',help='pgd attack alpha')
parser.add_argument('-steps','--steps',default=7,type=int,metavar='N',help='pgd attack steps')
# parser.add_argument('-bb','--bbmodel',default='vgg11_clean_l2[0.000500]bb',type=str,help='black box model') #
parser.add_argument('-bb','--bbmodel', default=0, type=float, metavar='N', help='black box model')

parser.add_argument('-stdout','--stdout',default='',type=str,help='log file')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args
    if args.dataset.lower() == 'cifar10':
        use_cifar10 = True
        num_labels = 10
        init_c = 3
    elif args.dataset.lower() == 'cifar100':
        use_cifar10 = False
        num_labels = 100
        init_c = 3
    elif args.dataset.lower() == 'svhn':
        num_labels = 10
        init_c = 3
    elif args.dataset.lower() == 'gesture':
        num_labels = 11
        init_c = 2

    # >>>>>>>IMPORTANT<<<<<<<< Edit log_dir
    log_dir = '%s-results' % (args.dataset)
    if args.seed_in_dir == 1:
        log_dir += '-sd%d' % args.seed
    model_dir = '%s-checkpoints' % (args.dataset)
    model_baseline_dir = '%s-baseline-checkpoints' % (args.dataset)
    model_static_dir = '%s-static-checkpoints' % (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed_all(args.seed)
    if 'cifar' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_cifar(use_cifar10=use_cifar10)
    elif args.dataset.lower() == 'svhn':
        train_dataset, val_dataset, znorm = data_loaders.build_svhn()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers, pin_memory=True)
        

    if 'vgg' in args.model.lower():
        model = VGG(args.model.lower(), args.time, num_labels, znorm, init_c=init_c)
        args.bbmodel = 'vgg11_clean_l2[0.000500]bb'
    elif 'wideresnet' in args.model.lower():
        model = WideResNet(args.model.lower(), args.time, num_labels, znorm, init_c=init_c)
        args.bbmodel = 'wideresnet16_clean_l2[0.000500]bb'
    else:
        raise AssertionError("model not supported")

    model.set_simulation_time(args.time)
    
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

    if len(args.bbmodel) > 0:
        bbmodel = copy.deepcopy(model)

    if args.allow_sparse:
        model = replace(model)
    

    model.eval()
    if 'cifar' in args.dataset.lower():
        model(torch.rand(5, 3, 32, 32).to(device))
    else:
        pass
        model(torch.rand(16, 10, 2, 128, 128).to(device))
    model.to(device)

    if os.path.exists(os.path.join(model_dir, args.identifier + args.suffix + '.pth')):
        state_dict = torch.load(os.path.join(model_dir, args.identifier + args.suffix + '.pth'))
    elif os.path.exists(os.path.join(model_baseline_dir, args.identifier + args.suffix + '.pth')):
        state_dict = torch.load(os.path.join(model_baseline_dir, args.identifier + args.suffix + '.pth'))
    elif os.path.exists(os.path.join(model_static_dir, args.identifier + args.suffix + '.pth')):
        state_dict = torch.load(os.path.join(model_static_dir, args.identifier + args.suffix + '.pth'))
    model.load_state_dict(state_dict)

    # have bb model
    if len(args.bbmodel) > 0:
        # bbmodel = copy.deepcopy(model)
        if os.path.exists(os.path.join(model_dir, args.bbmodel + '.pth')):
            bbstate_dict = torch.load(os.path.join(model_dir, args.bbmodel + '.pth'))
        elif os.path.exists(os.path.join(model_baseline_dir, args.bbmodel + '.pth')):
            bbstate_dict = torch.load(os.path.join(model_baseline_dir, args.bbmodel + '.pth'))
        elif os.path.exists(os.path.join(model_static_dir, args.bbmodel + '.pth')):
            bbstate_dict = torch.load(os.path.join(model_static_dir, args.bbmodel + '.pth'))
        bbmodel.load_state_dict(bbstate_dict, strict=False)
    else:
        bbmodel = None

    if len(args.config) > 0:
        with open(args.config+'.json', 'r') as f:
            config = json.load(f)
    else:
        config = [{}]
    
    # IMPORTANT<<<<<<<<<<<<< modifed
    identifier = args.identifier + args.suffix
    logger = get_logger(os.path.join(log_dir, '%s.log' % (identifier)))
    logger.info('start testing!')

    for atk_config in config:
        logger.info(json.dumps(atk_config))
        for arg in atk_config.keys():
            setattr(args, arg, atk_config[arg])
        if 'bb' in atk_config.keys() and atk_config['bb']:
            atkmodel = copy.deepcopy(bbmodel)
        else:
            atkmodel = copy.deepcopy(model)

        ff = BPTT_attack # default
        if 'cifar' in args.dataset.lower():
            rpst = lambda x: x
        else:
            rpst = represent_module

        identity = lambda x: x
        if args.attack.lower() == 'fgsm':
            atk = attack.FGSM(atkmodel, forward_function = ff, eps = args.eps / 255, targeted=(args.target==1),
                loss_function = nn.CrossEntropyLoss(), representation = identity, T = args.time)
        elif args.attack.lower() == 'fgml2':
            atk = attack.FGML2(atkmodel, forward_function = ff, eps = args.eps / 255, targeted=(args.target==1),
                loss_function = nn.CrossEntropyLoss(), representation = identity, T = args.time)
        elif args.attack.lower() == 'pgd':
            atk = attack.PGD(atkmodel, forward_function=ff, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps, targeted=(args.target==1),
                            loss_function = nn.CrossEntropyLoss(), representation = identity, T=args.time)
        elif args.attack.lower() == 'pgdl2':
            atk = attack.PGDL2(atkmodel, forward_function=ff, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps, targeted=(args.target==1),
                            loss_function = nn.CrossEntropyLoss(), representation = identity, T=args.time)
        else:
            atk = None
        
        acc, success_rate = val(model, test_loader, device, rpst, args.time, atk=atk, num_targets=num_labels)
        logger.info('Robust acc\t acc={:.3f} success_rate={:.3f}'.format(acc, success_rate))


if __name__ == "__main__":
    main()