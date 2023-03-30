from model import resnet18
import torch
from utils import data as dataset
import torch.nn as nn
import numpy as np
from torch.optim import SGD


model = resnet18.ResNet18(num_classes=100).cuda()
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
train_loader, valid_loader, test_loader, \
    test_onehot, test_label = dataset.get_loader('cifar100', '/home/coco/datasets/', 128)
cls_criterion = nn.CrossEntropyLoss().cuda()

model.train()

model_params = [m for m, _ in model.named_parameters()]

for i, (input, target, idx) in enumerate(train_loader):
    mask = np.random.binomial(1, 0.1, len(model_params))
    frozen = [elem for keep, elem in zip(mask, model_params) if keep]
    print(frozen)
    previous = {}
    for m, p in model.named_parameters():
        if m in frozen:
            p.grad = None
            p.requires_grad = False
            previous[m] = p.detach().clone().cpu()
    output = model(input.cuda())
    loss = cls_criterion(output, target.cuda())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for m, p in model.named_parameters():
        if m in frozen:
            assert torch.equal(p.detach().cpu(), previous[m]), f'values modified of the frozen layer {m}'
            p.grad = torch.zeros_like(p)
            p.requires_grad = True