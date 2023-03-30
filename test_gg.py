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
prev_gradients = None
last_step = None
for i, (input, target, idx) in enumerate(train_loader):
    output = model(input.cuda())
    loss = cls_criterion(output, target.cuda())
    optimizer.zero_grad()
    loss.backward()
    if prev_gradients is not None:
        temp_grads = []
        for idx, (m, p) in enumerate(model.named_parameters()):
            print(f'{m}, {torch.abs(p.grad-prev_gradients[idx]).sum()}')
            temp_grads.append(p.grad.detach())
            grad_mask = ((p.grad * prev_gradients[idx]) > 0).float()
            p.grad = grad_mask * (p.grad + prev_gradients[idx])
        prev_gradients = temp_grads
        optimizer.step()
    else:
        prev_gradients = [p.grad.detach() for m, p in model.named_parameters()]

    if i == 10:
        break