"""PyTorch port of YDaiLab PopPhy.py/config.py (see README for deviations)."""
import torch
from torch import nn

class PopPhyCNN(nn.Module):
    def __init__(self, shape, classes):
        super().__init__()
        h,w = shape
        if h < 4 or w < 11: raise ValueError('Tree matrix too small for author valid convolution/pooling')
        self.conv = nn.Conv2d(1,32,(3,10))
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(32*((h-2)//2)*((w-9)//2),32)
        self.output = nn.Linear(32,classes)
        # Keras defaults: Glorot uniform kernels and zero biases.
        for layer in (self.conv,self.fc,self.output):
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self,x):
        if self.training: x = x + torch.randn_like(x)*0.01
        x = self.pool(torch.relu(self.conv(x))).flatten(1)
        x = self.drop(x)
        x = self.drop(torch.relu(self.fc(x)))
        return self.output(x)

"""Genus adaptation of Wang et al. DeepPhylo equations (2)--(6)."""
import torch
from torch import nn

class DeepPhylo(nn.Module):
    def __init__(self, embeddings, classes, hidden=64, kernel=3):
        super().__init__()
        self.register_buffer('embeddings',torch.as_tensor(embeddings,dtype=torch.float32))
        self.abundance = nn.Linear(len(embeddings),hidden)
        self.conv = nn.Conv1d(embeddings.shape[1],hidden,kernel)
        self.output = nn.Linear(hidden,classes)
        self.kernel = kernel

    def forward(self, abundance, present):
        # Compact only observed taxa, preserving fixed genus order. Padding is
        # masked after convolution, including bias; it cannot win max pooling.
        lengths = present.sum(1)
        width = max(self.kernel,int(lengths.max().item()))
        sequence = abundance.new_zeros((len(abundance),self.embeddings.shape[1],width))
        for b,mask in enumerate(present):
            e = self.embeddings[mask]
            sequence[b,:,:len(e)] = e.T
        features = torch.relu(self.conv(sequence))
        valid_lengths = (lengths-self.kernel+1).clamp(min=1)
        mask = torch.arange(features.shape[2],device=features.device)[None,:] < valid_lengths[:,None]
        features = features.masked_fill(~mask[:,None,:],-torch.inf).amax(2)
        features = torch.where((lengths>0)[:,None],features,torch.zeros_like(features))
        return self.output(torch.relu(self.abundance(abundance))*features)
