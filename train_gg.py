import utils.crl_utils
from utils import utils
import time
import numpy as np
import torch


def train(loader, model, criterion, optimizer, epoch, logger, prev_gradients=None):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    end = time.time()
    model.train()
    for i, (input, target, _) in enumerate(loader):
        data_time.update(time.time() - end)
        input, target = input.cuda(), target.long().cuda()
        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()

        if prev_gradients is not None:
            temp_grads = []
            for idx, (m, p) in enumerate(model.named_parameters()):
                temp_grads.append(p.grad.detach())
                grad_mask = ((p.grad * prev_gradients[idx]) > 0).float()
                p.grad = grad_mask * (p.grad + prev_gradients[idx])
            prev_gradients = temp_grads
            optimizer.step()
        else:
            prev_gradients = [p.grad.detach() for m, p in model.named_parameters()]

        prec, correct = utils.accuracy(output, target)
        total_losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

    logger.write([epoch, total_losses.avg, top1.avg])
    return {'loss': total_losses.avg, 'acc': top1.avg}, prev_gradients
