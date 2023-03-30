import utils.crl_utils
from utils import utils
import time
import numpy as np
import torch


def train(loader, model, criterion, optimizer, epoch, logger, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    end = time.time()
    model.train()
    model_params = list(set([m.split('.')[0] for m, _ in model.named_parameters()]))
    for i, (input, target, _) in enumerate(loader):
        mask = np.random.binomial(1, args.scale, len(model_params))
        frozen = [elem for keep, elem in zip(mask, model_params) if keep]
        for m, p in model.named_parameters():
            if m.split('.')[0] in frozen:
                p.grad = None
                p.requires_grad = False

        data_time.update(time.time() - end)
        input, target = input.cuda(), target.long().cuda()
        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prec, correct = utils.accuracy(output, target)
        total_losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        for m, p in model.named_parameters():
            if m.split('.')[0] in frozen:
                p.grad = torch.zeros_like(p)
                p.requires_grad = True

        batch_time.update(time.time() - end)
        end = time.time()

    logger.write([epoch, total_losses.avg, top1.avg])
    return {'loss': total_losses.avg, 'acc': top1.avg}
