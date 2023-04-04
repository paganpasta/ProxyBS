import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import argparse
import os
import csv
import resource

from model import LeNet
from model import AlexNet
from model import resnet
from model import resnet18
from model import densenet_BC
from model import vgg
from model import mobilenet
from model import efficientnet
from model import wrn
from model import convmixer
from utils import data as dataset
from utils import metrics
from utils import utils
import train_val



parser = argparse.ArgumentParser(description='Rethinking CC for FP')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--plot', default=20, type=int, help='')
parser.add_argument('--data', default='cifar10', type=str, help='Dataset name to use [cifar10, cifar100]')
parser.add_argument('--model', default='res110', type=str,
                    help='Models name to use [res110, dense, wrn, cmixer, efficientnet, mobilenet, vgg]')
parser.add_argument('--method', default='val', type=str, help='[val]')
parser.add_argument('--data_path', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--load_path', default='./output/', type=str, help='Savefiles directory')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--print-freq', '-p', default=200, type=int, metavar='N', help='print frequency (default: 10)')

args = parser.parse_args()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True
    args.distributed = False

    train_loader, valid_loader, test_loader, \
        test_onehot, test_label = dataset.get_loader(args.data, args.data_path, args.batch_size, args)

    if args.data == 'cifar100':
        num_class = 100
        args.classnumber = 100
    else:
        num_class = 10
        args.classnumber = 10

    model_dict = {
        "num_classes": num_class,
    }
    if args.model == 'resnet18':
        model = resnet18.ResNet18(**model_dict).cuda()
    elif args.model == 'res110':
        model = resnet.resnet110(**model_dict).cuda()
    elif args.model == 'dense3':
        model = densenet_BC.DenseNet3(depth=100, num_classes=num_class,
                                      growth_rate=12, reduction=0.5,
                                      bottleneck=True, dropRate=0.0).cuda()
    elif args.model == 'vgg16':
        model = vgg.vgg16(**model_dict).cuda()
    elif args.model == 'wrn28':
        model = wrn.WideResNet(28, num_class, 10).cuda()
    elif args.model == 'efficientnet':
        model = efficientnet.efficientnet(**model_dict).cuda()
    elif args.model == 'mobilenet':
        model = mobilenet.mobilenet(**model_dict).cuda()
    elif args.model == "cmixer":
        model = convmixer.ConvMixer(256, 16, kernel_size=8, patch_size=1, n_classes=num_class).cuda()
    elif args.model == 'LeNet':
        model = LeNet.LeNet().cuda()
    elif args.model == 'AlexNet':
        model = AlexNet.AlexNet().cuda()

    cls_criterion = nn.CrossEntropyLoss().cuda()
    try:
        model.load_state_dict(torch.load(args.load_path)) 
    except:
        state_dict = torch.load(args.load_path)
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' in k:
                name = k[7:] # remove `module.`
                new_dict[name] = v
        model.load_state_dict(new_dict, strict=True)

    scores = metrics.calc_metrics(args, test_loader, test_label, test_onehot, model, cls_criterion)


if __name__ == "__main__":
    main()
