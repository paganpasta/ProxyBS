import utils.crl_utils
from utils import utils
import time
from utils.val_utils import ValSmoothing
import torch
import torch.nn.functional as F


def train(loader, model, criterion, optimizer, epoch, logger, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    end = time.time()
    model.train()

    for i, (input, target, idx) in enumerate(loader):
        data_time.update(time.time() - end)
        input, target = input.cuda(), target.long().cuda()

        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        if args.method == 'sam' or args.method == 'fmfp':
            optimizer.first_step(zero_grad=True)
            loss_again = criterion(model(input), target)
            loss_again.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        # record loss and accuracy
        prec, correct = utils.accuracy(output, target)

        total_losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.write([epoch, total_losses.avg, top1.avg])
    return {'loss': total_losses.avg, 'acc': top1.avg}


def train_val(loader, val_loader, cls_scores, model, criterion, optimizer, epoch, logger, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    cls_losses = utils.AverageMeter()
    val_losses = utils.AverageMeter()
    end = time.time()
    model.train()

    criterion_val = ValSmoothing()
    val_iter = iter(val_loader)

    for i, (input, target, idx) in enumerate(loader):
        data_time.update(time.time() - end)
        input, target = input.cuda(), target.long().cuda()

        output = model(input)
        cls_loss = criterion(output, target)
        val_targets = cls_scores.get(target)
        val_loss = criterion_val(output, val_targets)
        loss = cls_loss + val_loss * args.alpha
        optimizer.zero_grad()
        loss.backward()
        if args.method in ['sam', 'fmfp', 'fmfp_val', 'sam_val']:
            optimizer.first_step(zero_grad=True)
            output_again = model(input)
            loss_again = criterion(output_again, target) + args.val_weight * criterion_val(output, val_targets)
            loss_again.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        # record loss and accuracy
        prec, correct = utils.accuracy(output, target)

        cls_losses.update(cls_loss.item(), input.size(0))
        val_losses.update(val_loss.item(), input.size(0))
        total_losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        if 'val' in args.method:
            try:
                input, target, idx = next(val_iter)
            except:
                val_iter = iter(val_loader)
                input, target, idx = next(val_iter)
            model.eval()
            with torch.no_grad():
                output = F.softmax(model(input.cuda()), dim=1)
                cls_scores.update(output.detach(), target.long().cuda())
            model.train()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.write([epoch, total_losses.avg, top1.avg])
    return {'loss': total_losses.avg, 'acc': top1.avg, 'cls_loss': cls_losses.avg, 'val_loss': val_losses.avg}
