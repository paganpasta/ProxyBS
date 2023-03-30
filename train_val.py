import utils.crl_utils
from utils import utils
import torch.nn as nn
import time
import torch
import torch.nn.functional as F


class ValSmoothing(nn.Module):
    """NLL loss with val smoothing.
    """

    def __init__(self):
        super(ValSmoothing, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, xs, targets):
        logprobs = F.log_softmax(xs, dim=-1)
        return self.kl_div(logprobs, targets)


def train(loader, val_loader, cls_scores, model, criterion, optimizer, epoch, logger, args):
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
        if args.method == 'val':
            cls_loss = criterion(output, target)
            val_targets = cls_scores.get(target)
            val_loss = criterion_val(output, val_targets)
            loss = cls_loss + val_loss * args.alpha
            cls_losses.update(cls_loss.item(), input.size(0))
            val_losses.update(val_loss.item(), input.size(0))
        else:
            raise NotImplementedError(f'{args.method} is not supported!')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec, correct = utils.accuracy(output, target)
        total_losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        if args.method == 'val':
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

        batch_time.update(time.time() - end)
        end = time.time()

    logger.write([epoch, total_losses.avg, top1.avg])
    return {'loss': total_losses.avg, 'acc': top1.avg, 'cls_loss': cls_losses.avg, 'val_loss': val_losses.avg}
