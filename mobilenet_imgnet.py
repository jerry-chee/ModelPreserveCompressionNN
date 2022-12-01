from turtle import forward, up
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import scipy
import id_utils
import nn_utils

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()
        #self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        #out = F.avg_pool2d(out, 2)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



class BlockPruned(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    @torch.no_grad()
    def __init__(self, in_planes, out_planes, original=None, stride=1, out=None, 
                conv=None, bn=None, k=1):
        super(BlockPruned, self).__init__()
        self.conv1 = deepcopy(original.conv1)#nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = deepcopy(original.bn1)#nn.BatchNorm2d(in_planes)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        #Z=F.relu(self.bn1(self.conv1(out)))
        Z = self.forward_block_v1(out, blocksize=100)
        fi = Z.shape[1]
        Zr = Z.permute(0,2,3,1).reshape(-1,fi).cpu().detach().numpy()

        fp=int(Zr.shape[1]*k)

        (k_idx,T) = id_utils.getID(fp, Zr)
            
        #fix up the next layer.   
        T = torch.Tensor(T).to(self.device)
        Wnext=original.conv2.weight.clone()
        Wnext = Wnext.permute(0,2,3,1)
        Wnext = torch.matmul(Wnext, T.T)
        Wnext = Wnext.permute(0,3,1,2)
        self.conv2 = nn.Conv2d(fp, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2.weight=nn.Parameter(Wnext, requires_grad=True)#nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = deepcopy(original.bn2)#nn.BatchNorm2d(out_planes)
        # Sub-select current layer
        self.conv2.in_channels=fp
        self.conv1.weight = nn.Parameter(self.conv1.weight[k_idx,:].clone(), requires_grad=True)
        self.conv1.out_features = fp
        self.conv1.in_features=fp
        self.conv1.groups=fp

        #sub-select previous layer 
        conv.weight = nn.Parameter(conv.weight[k_idx,:].clone(), requires_grad=True)
        conv.out_features=fp
        #sub-select previous BN.  
        bn.weight = nn.Parameter(bn.weight[k_idx].clone(),requires_grad=True)
        bn.bias = nn.Parameter(bn.bias[k_idx].clone(),requires_grad=True)
        bn.register_buffer('running_mean', bn.running_mean[k_idx].clone())
        bn.register_buffer('running_var',bn.running_var[k_idx].clone())
        bn.num_features = fp

        #sub-select this layer's bn
        self.bn1.weight = nn.Parameter(self.bn1.weight[k_idx].clone(),requires_grad=True)
        self.bn1.bias = nn.Parameter(self.bn1.bias[k_idx].clone(),requires_grad=True)
        self.bn1.register_buffer('running_mean', self.bn1.running_mean[k_idx].clone())
        self.bn1.register_buffer('running_var',self.bn1.running_var[k_idx].clone())
        self.bn1.num_features = fp

    @torch.no_grad()
    def forward_block_v1(self, X, blocksize):
        out_ls = []
        for X_i in torch.split(X, blocksize, dim=0):
            out_i = F.relu(self.bn1(self.conv1(X_i.to(self.device)))).cpu()
            out_ls.append(out_i)
        out = torch.cat(out_ls)
        return out
    
    def forward(self, x):
        x=self.conv1(x)
        out = F.relu(self.bn1(x))  
        out = F.relu(self.bn2(self.conv2(out)))
        return out



class MobileNetPruned_ID(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    @torch.no_grad()
    def __init__(self, original, X_prune, num_classes=1000, k=1.0):
        super(MobileNetPruned_ID, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.k = k
        self.conv1 = deepcopy(original.conv1)#nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        #X_prune, y_prune = next(iter(prune_loader))
        
        self.bn1 = deepcopy(original.bn1)#nn.BatchNorm2d(32)
        #out = F.relu(self.bn1(self.conv1(X_prune)))
        out = self.forward_block_v1(X_prune, blocksize=100)
        self.layers = self._make_layers(in_planes=32, original=original, out=out)
        self.linear = deepcopy(original.linear)#nn.Linear(1024, num_classes)

    @torch.no_grad()
    def _make_layers(self, in_planes, original=None, out=None):
        layers = []
        prevConv=self.conv1
        prevBN=self.bn1
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            #temp=original.layers[len(layers)](out)
            temp = self.forward_block_v2(original, layers, out, blocksize=100)
            block=BlockPruned(in_planes, out_planes, original.layers[len(layers)], 
                                stride, out=out, conv=prevConv, bn=prevBN, k=self.k)
            prevConv=block.conv2
            prevBN=block.bn2
            out=temp
            layers.append(block)
            in_planes = out_planes
        return nn.Sequential(*layers)

    @torch.no_grad()
    def forward_block_v1(self, X, blocksize):
        out_ls = []
        for X_i in torch.split(X, blocksize, dim=0):
            out_i = F.relu(self.bn1(self.conv1(X_i.to(self.device)))).cpu()
            out_ls.append(out_i)
        out = torch.cat(out_ls)
        return out

    @torch.no_grad()
    def forward_block_v2(self, og, layers, X, blocksize):
        out_ls = []
        for X_i in torch.split(X, blocksize, dim=0):
            out_i = og.layers[len(layers)](X_i.to(self.device)).cpu()
            out_ls.append(out_i)
        out = torch.cat(out_ls)
        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        #out = F.avg_pool2d(out, 2)
        ##out = out.view(out.size(0), -1)
        #out = out.resize(out.size(0), -1)
        out = out.mean([2,3])
        out = self.linear(out)
        return out

class MobileNetPruned_IDiter(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    @torch.no_grad()
    def __init__(self, original, X_prune, k, num_classes=1000):
        super(MobileNetPruned_IDiter, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.k = k
        self.conv1 = deepcopy(original.conv1)#nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        #X_prune, y_prune = next(iter(prune_loader))
        
        self.bn1 = deepcopy(original.bn1)#nn.BatchNorm2d(32)
        #out = F.relu(self.bn1(self.conv1(X_prune)))
        out = self.forward_block_v1(X_prune, blocksize=100)
        self.layers = self._make_layers(in_planes=32, original=original, out=out)
        self.linear = deepcopy(original.linear)#nn.Linear(1024, num_classes)

    @torch.no_grad()
    def _make_layers(self, in_planes, original, out=None):
        layers = []
        prevConv=self.conv1
        prevBN=self.bn1
        holder=deepcopy(out)
        scores=[]
        
        ms = (nn_utils.model_summary(original, summary_input=(3,224,224), input_res=224))[1]
        for x in range(len(self.cfg)):

            layer=original.layers[x]
            #Z=F.relu(layer.bn1(layer.conv1(holder)))
            Z = self.forward_block_v1b(holder, layer, blocksize=100)
            fi = Z.shape[1]
            Zr = Z.permute(0,2,3,1).reshape(-1,fi)
            fp=int(Zr.shape[1]*self.k)
            ######## SVD
            #svd=np.linalg.svd(Zr,full_matrices=False, compute_uv=False)
            #print(svd[fp], svd[0], ms[2*x+1], svd[fp]/svd[0]/ms[2*x+1])
            #scores.append(svd[fp]/svd[0]/ms[2*x+1])
            ########
            ######## QR
            R, _ = scipy.linalg.qr(Zr, mode='r', pivoting=True)
            #print(R[fp,fp], R[0,0], ms[2*x+1], R[fp,fp]/R[0,0]/ms[2*x+1])
            scores.append(np.abs(R[fp,fp]/R[0,0]/ms[2*x+1]))
            ########
            #holder=original.layers[x](holder)
            holder = self.forward_block_v1c(holder, original.layers[x], blocksize=100)
        print(scores)
        layer=np.argmin(scores)
        print("PRUNING layer:")
        print(layer)
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            #temp=original.layers[len(layers)](out)
            temp = self.forward_block_v2(original, layers, out, blocksize=100)
            if len(layers)==layer:
                block=BlockPruned(in_planes, out_planes, original.layers[len(layers)], stride,out=out , conv=prevConv, bn=prevBN, k=self.k)
            else:
                block=deepcopy(original.layers[len(layers)])
            prevConv=block.conv2
            prevBN=block.bn2
            with torch.no_grad():
                out=temp
            layers.append(block)
            in_planes = out_planes
        return nn.Sequential(*layers)

    @torch.no_grad()
    def forward_block_v1(self, X, blocksize):
        out_ls = []
        for X_i in torch.split(X, blocksize, dim=0):
            out_i = F.relu(self.bn1(self.conv1(X_i.to(self.device)))).cpu()
            out_ls.append(out_i)
        out = torch.cat(out_ls)
        return out

    @torch.no_grad()
    def forward_block_v1b(self, X, layer, blocksize):
        out_ls = []
        for X_i in torch.split(X, blocksize, dim=0):
            out_i = F.relu(layer.bn1(layer.conv1(X_i.to(self.device)))).cpu()
            out_ls.append(out_i)
        out = torch.cat(out_ls)
        return out

    @torch.no_grad()
    def forward_block_v1c(self, X, fn, blocksize):
        out_ls = []
        for X_i in torch.split(X, blocksize, dim=0):
            out_i = fn(X_i.to(self.device)).cpu()
            out_ls.append(out_i)
        out = torch.cat(out_ls)
        return out

    @torch.no_grad()
    def forward_block_v2(self, og, layers, X, blocksize):
        out_ls = []
        for X_i in torch.split(X, blocksize, dim=0):
            out_i = og.layers[len(layers)](X_i.to(self.device)).cpu()
            out_ls.append(out_i)
        out = torch.cat(out_ls)
        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        #out = F.avg_pool2d(out, 2)
        #out = out.view(out.size(0), -1)
        out = out.mean([2,3])
        out = self.linear(out)
        return out

def mobilenet_IDiter(args, model, prune_loader):
    sizes=[]
    acc=[]
    pruned=MobileNetPruned_IDiter(original=model)
    for i in range(0, 100):
        pruned=MobileNetPruned_IDiter(original=pruned)
        acc.append(nn_utils.test(args, pruned, 'cpu', prune_loader, criterion = nn.CrossEntropyLoss(), epoch=160, returnAcc=True))
        print(acc)
        ms, conv, lin= nn_utils.model_summary(pruned,summary_input=(3,224,224), input_res=32)
        sizes.append(ms)

