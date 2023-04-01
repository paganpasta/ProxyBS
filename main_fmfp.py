import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import os
import csv
import wandb

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
from utils import val_utils
import train_fmfp

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.sam import SAM


def csv_writter(path, dic, start):
    if os.path.isdir(path) == False: os.makedirs(path)
    os.chdir(path)
    # Write dic
    if start == 1:
        mode = 'w'
    else:
        mode = 'a'
    with open('logs.csv', mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        if start == 1:
            writer.writerow(dic.keys())
        writer.writerow([elem["string"] for elem in dic.values()])


class Counter(dict):
    def __missing__(self, key):
        return None


parser = argparse.ArgumentParser(description='Rethinking CC for FP')
parser.add_argument('--epochs', default=201, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--plot', default=20, type=int, help='')
parser.add_argument('--data', default='cifar10', type=str, help='Dataset name to use [cifar10, cifar100]')
parser.add_argument('--model', default='wrn28', type=str,
                    help='Models name to use [res110, dense, wrn28, cmixer, efficientnet, mobilenet, vgg]')
parser.add_argument('--method', default='fmfp', type=str, help='val_+[sam, swa, fmfp]')
parser.add_argument('--data_path', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--save_path', default='./output/flat/', type=str, help='Savefiles directory')
parser.add_argument('--val_weight', default=0.25, type=float, help='Val loss weight')
parser.add_argument('--gamma', default=0.1, type=float, help='CLS SCORE DECAY')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--print-freq', '-p', default=200, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--group', default='DEBUG', type=str, help='wandb group to log')
args = parser.parse_args()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True
    args.distributed = False

    args.save_path = os.path.join(args.save_path, f'{args.data}_{args.model}_{args.method}_{args.val_weight}')
    os.makedirs(args.save_path, exist_ok=True)

    wandb_logger = utils.init_wandb(args)

    train_loader, valid_loader, test_loader, \
        test_onehot, test_label = dataset.get_loader(args.data, args.data_path, args.batch_size, args)

    if args.data == 'cifar100':
        num_class = 100
    else:
        num_class = 10
    model_dict = {
        "num_classes": num_class,
    }
    args.num_class = num_class
    
    print(torch.cuda.is_available(), torch.cuda.device_count())

    if args.model == 'resnet18':
        model = resnet18.ResNet18(**model_dict).cuda()
    elif args.model == 'res110':
        model = resnet.resnet110(**model_dict).cuda()
    elif args.model == 'dense':
        model = densenet_BC.DenseNet3(depth=100, num_classes=num_class,
                                      growth_rate=12, reduction=0.5,
                                      bottleneck=True, dropRate=0.0).cuda()
    elif args.model == 'vgg':
        model = vgg.vgg16(**model_dict).cuda()
    elif args.model == 'wrn28':
        model = wrn.WideResNet(28, num_class, 10).cuda()
    elif args.model == 'efficientnet':
        model = efficientnet.efficientnet(**model_dict).cuda()
    elif args.model == 'mobilenet':
        model = mobilenet.mobilenet(**model_dict).cuda()
    elif args.model == "cmixer":
        model = convmixer.ConvMixer(256, 16, kernel_size=8, patch_size=1, n_classes=num_class).cuda()

    wandb.watch(model, log='all', log_freq=500)
    cls_criterion = nn.CrossEntropyLoss().cuda()
    # make logger
    train_logger = utils.Logger(os.path.join(args.save_path, 'train.log'))
    result_logger = utils.Logger(os.path.join(args.save_path, 'result.log'))

    base_lr = 0.1  # Initial learning rate
    lr_strat = [80, 130, 170]
    lr_factor = 0.1  # Learning rate decrease factor
    custom_weight_decay = 5e-4  # Weight Decay
    custom_momentum = 0.9  # Momentum

    if 'sam' in args.method:
        if args.model == "convmixer":
            base_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=custom_weight_decay)
            optimizer = SAM(model.parameters(), base_optimizer)
            scheduler = None
        else:
            base_optimizer = torch.optim.SGD
            optimizer = SAM(model.parameters(), base_optimizer, lr=base_lr, momentum=custom_momentum,
                            weight_decay=custom_weight_decay)
            scheduler = MultiStepLR(optimizer, milestones=lr_strat, gamma=lr_factor)

    elif 'swa' in args.method:
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=custom_momentum,
                                    weight_decay=custom_weight_decay)
        if args.model == "convmixer":
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=custom_weight_decay)
        swa_model = AveragedModel(model)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        swa_start = 120
        swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    elif 'fmfp' in args.method:
        if args.model == "convmixer":
            base_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=custom_weight_decay)
            optimizer = SAM(model.parameters(), base_optimizer)
        else:
            base_optimizer = torch.optim.SGD
            optimizer = SAM(model.parameters(), base_optimizer, lr=base_lr, momentum=custom_momentum,
                            weight_decay=custom_weight_decay)
        swa_model = AveragedModel(model)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        swa_start = 120
        swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    # initialise val scores
    cls_scores = val_utils.get_val_scores(model, valid_loader, num_classes=args.num_class, gamma=args.gamma)

    # start Train
    for epoch in range(1, args.epochs + 1):
        if 'val' not in args.method:
            training_metrics = train_fmfp.train(train_loader,
                                                model,
                                                cls_criterion,
                                                optimizer,
                                                epoch,
                                                train_logger,
                                                args)
        else:
            training_metrics = train_fmfp.train_val(train_loader, valid_loader, cls_scores,
                                                    model,
                                                    cls_criterion,
                                                    optimizer,
                                                    epoch,
                                                    train_logger,
                                                    args)

        if 'swa' in args.method or 'fmfp' in args.method:
            if scheduler != None:
                if epoch > swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    scheduler.step()
        else:
            if scheduler != None:
                scheduler.step()

        # calc measure
        if epoch % 10 == 1 or epoch == args.epochs:
            print(epoch, 100 * '#')
            scores = metrics.calc_metrics(args, test_loader, test_label, test_onehot, model, cls_criterion)
            wandb_logger.log({'val': scores, 'train': training_metrics}, step=epoch)

            ckpt = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'scores': scores,
            }
            torch.save(ckpt, os.path.join(args.save_path, 'model.pth'))

            # result write
            result_logger.write([scores['acc'], scores['auroc'], scores['aupr_s'],
                                 scores['aupr'], scores['fpr'], scores['tnr'],
                                 scores['aurc'], scores['eaurc'], scores['ece'],
                                 scores['nll'], scores['bs']])

    torch.save(model.state_dict(), os.path.join(args.save_path, 'last.pth'))
    # Update bn statistics for the swa_model at the end
    if 'swa' in args.method or 'fmfp' in args.method:
        torch.optim.swa_utils.update_bn(train_loader, swa_model.cpu())
        model = swa_model.cuda()
    torch.save(model.state_dict(), os.path.join(args.save_path, 'avg_model.pth'))

    acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, ece, nll, brier = metrics.calc_metrics(args, test_loader,
                                                                                                  test_label,
                                                                                                  test_onehot,
                                                                                                  model, cls_criterion)
    # result write
    result_logger.write([acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, ece, nll, brier])


if __name__ == "__main__":
    main()
