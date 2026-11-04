"""Shared float32 encodings and unpretrained ResNets."""
import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50

class BasicBlock1D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.projection = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)) if stride != 1 or cin != cout else nn.Identity()
    def forward(self, x):
        identity = self.projection(x)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)) + identity)

class ResNet18OneD(nn.Module):
    def __init__(self, classes):
        super().__init__()
        layers = [nn.Conv1d(1, 64, 7, 2, 3, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(3, 2, 1)]
        cin = 64
        for cout, stride in zip((64,128,256,512), (1,2,2,2)):
            layers.extend([BasicBlock1D(cin, cout, stride), BasicBlock1D(cout, cout)])
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, classes)
        for m in self.modules():
            if isinstance(m, nn.Conv1d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
    def forward(self, x):
        return self.fc(self.pool(self.layers(x)).flatten(1))

def build_model(encoding, depth, classes):
    if encoding == '1d': return ResNet18OneD(classes)
    model = {18: resnet18, 34: resnet34, 50: resnet50}[depth](weights=None, num_classes=classes)
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    return model

def encode(x, encoding, hilbert):
    if encoding == '1d': return x.unsqueeze(1)
    if encoding == 'hilbert': x = x[:, hilbert]
    return x.reshape(-1, 1, 64, 64)
