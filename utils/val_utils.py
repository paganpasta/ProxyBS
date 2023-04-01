import torch
import torch.nn.functional as F
import torch.nn as nn


class ValSmoothing(nn.Module):
    """NLL loss with val smoothing.
    """

    def __init__(self):
        super(ValSmoothing, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, xs, targets):
        logprobs = F.log_softmax(xs, dim=-1)
        return self.kl_div(logprobs, targets)


class ValScores(nn.Module):
    def __init__(self, num_classes, gamma):
        super().__init__()
        self.val_preds = torch.zeros(num_classes, num_classes).requires_grad_(False).cuda()
        self.gamma = gamma

    def update(self, preds, labels):
        # labels = [b]
        # preds = [b x C]
        for c in range(preds.shape[1]):
            mask = torch.where(labels == c, 1.0, 0.0)
            num_instances = mask.sum()
            if num_instances > 0:
                self.val_preds[c] = (1 - self.gamma) * ((mask.unsqueeze(1) * preds).sum(dim=0) / num_instances) + \
                                self.gamma * self.val_preds[c]

    def get(self, labels):
        return self.val_preds[labels]


def get_val_scores(model, valid_loader, num_classes, gamma):
    cls_scores = ValScores(num_classes=num_classes, gamma=gamma)
    with torch.no_grad():
        model.eval()
        print('Initialising the val scores before training')
        for i, (input, target, idx) in enumerate(valid_loader):
            output = F.softmax(model(input.cuda()), dim=1)
            cls_scores.update(output.detach(), target.long().cuda())
        for i in range(num_classes):
            print('CLS', i, cls_scores.val_preds[i])