from collections import OrderedDict
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock, Bottleneck

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


# def load_weights_sequential(target, source_state):
#     new_dict = OrderedDict()
#     for (k1, v1), (k2, v2) in zip(target.state_dict().items(), source_state.items()):
#         new_dict[k1] = v2
#     target.load_state_dict(new_dict)
from torchvision.models.resnet import ResNet


def resnet18(pretrained=False, ratio=1):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(pretrained=False, ratio=1):
    model = ResNet(BasicBlock, [3, 4, 6, 3], ratio=ratio)
    return model

def resnet50(pretrained=False, ratio=1):
    model = ResNet(Bottleneck, [3, 4, 6, 3], ratio=ratio)
    return model

def resnet101(pretrained=False, ratio=1):
    model = ResNet(Bottleneck, [3, 4, 23, 3], ratio=ratio)
    return model

def resnet152(pretrained=False, ratio=1):
    model = ResNet(Bottleneck, [3, 8, 36, 3], ratio=ratio)
    return model