import torch
import torch.nn as nn
import torch.nn.functional as F

class FC1(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, bias1, bias2):
        super(FC1, self).__init__()
        self.fc1         = nn.Linear(in_feat, hidden_feat, bias1)
        self.fc2         = nn.Linear(hidden_feat, out_feat, bias2)
        self.in_feat     = in_feat
        self.hidden_feat = hidden_feat
        self.out_feat    = out_feat
        self.bias1       = bias1
        self.layers=[self.fc1, self.fc2]
        self.biases=[bias1, bias2]
        ## different init
        #self.fc1.weight = nn.Parameter(torch.Tensor(self.fc1.weight.shape).uniform_(-1/31.62, +1/31.62))
        #self.fc2.weight = nn.Parameter(torch.Tensor(self.fc2.weight.shape).uniform_(-1/31.62, +1/31.62))
        #self.fc2.bias   = nn.Parameter(torch.Tensor(self.fc2.bias.shape).uniform_(-1/31.62, +1/31.62))

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)

        return out

    def getZ(self, x):
        # non-linear output for ID
        out = self.fc1(x)
        out = F.relu(out)
        return out


class FC2(nn.Module):
    def __init__(self, feats, biases):
        super(FC2, self).__init__()
        if len(feats) != 4 or len(biases) != 3:
            raise ValueError('FC2(feats or biases lengths incorrect)')
        self.fc1    = nn.Linear(feats[0], feats[1], biases[0])
        self.fc2    = nn.Linear(feats[1], feats[2], biases[1])
        self.fc3    = nn.Linear(feats[2], feats[3], biases[1])
        self.feats  = feats
        self.biases = biases
        self.layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out
    def get_layers(self):
        return self.layers
    def getZ(self, x, layer):
        return F.relu(self.layers[layer](x))
    
