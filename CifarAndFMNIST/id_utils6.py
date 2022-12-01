import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np
import scipy.linalg
import nn_utils
import numpy.linalg as ln
import matplotlib
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import time
from train import model_summary

from torch.utils.data import TensorDataset, DataLoader
def getID(k, Z, W=None, mode='kT'):
    '''
    calculates ID 
    mode: kT, WT (what to return)
    requires (d, n) format
    currently in numpy because pytorch doesn't support QR pivots
    k: number of columns
    Z: layer after nonlinearity
    ''' 

    assert(k <= Z.shape[1])


    R, P = scipy.linalg.qr((Z), mode='r', pivoting=True)

    
    vals=np.linspace(0, Z.shape[1], 10)
    norms=[]

    #pickle.dump(s, open("figs/{}prunesize100.p".format(time.time()), 'wb'))
    if W is not None: Wk = W[:, P[0:k]]
    T = np.concatenate((
        np.identity(k),
        np.linalg.pinv(R[0:k, 0:k]) @ R[0:k, k:None]
        ), axis=1)
    T = T[:, np.argsort(P)]
    if mode == 'kT':
        return P[0:k], T
    elif mode == 'WT' and W is not None:
        return Wk, T
    else:
        raise NotImplementedError

def pruneIDLayer(layer, nextlayer, Z, k, ptype='FC-FC', extralayer=None, ft=False, input=None):
    '''
    ID pruning on specified layer, propagating interpolation matrix to next
    Note: Modifies In Place
    layer: old layer
    nextlayer: old next layer
    prunelayer: new pruned layer
    prunenextlayer: new pruned next layer
    select: number of neurons, or (TODO accuarcy)
    ptype: FC-FC, Conv-FC, Conv-Conv
    '''
    biasbool = layer.bias is not None
    print(ptype)
    with torch.no_grad():
        # select neurons and calculate interpolation
        # assign in place weight (and bias)
        k_idx, T = getID(k, Z, mode='kT')
        layer.weight = nn.Parameter(layer.weight[k_idx,:].clone(), 
                requires_grad=True)
        if biasbool:
            layer.bias = nn.Parameter(layer.bias[k_idx].clone(), 
                    requires_grad=True)
        T = torch.Tensor(T)
        
        # propagation to next layer
        if ptype == 'FC-FC':
            nextlayer.weight = nn.Parameter(nextlayer.weight @ T.T,
                    requires_grad=True)
            # don't do anything for bias
            # update meta info
            layer.out_features = k
            nextlayer.in_features = k
        elif ptype == 'FC-BN-FC':
            nextlayer.weight = nn.Parameter(nextlayer.weight @ T.T,
                    requires_grad=True)
            # don't do anything for bias
            # BN
            extralayer.weight = nn.Parameter(extralayer.weight[k_idx].clone(),
                    requires_grad=True)
            extralayer.bias = nn.Parameter(extralayer.bias[k_idx].clone(),
                    requires_grad=True)
            extralayer.register_buffer('running_mean', 
                    extralayer.running_mean[k_idx].clone())
            extralayer.register_buffer('running_var',
                    extralayer.running_var[k_idx].clone())
            # update meta info
            layer.out_features = k
            nextlayer.in_features = k
            extralayer.num_features = k
        elif ptype == 'Conv-FC':
            n = int(nextlayer.in_features / layer.out_channels)
            T = torch.kron(T.contiguous(), torch.eye(n))
            nextlayer.weight = nn.Parameter(nextlayer.weight @ T.T,
                    requires_grad=True)
            # don't do anything for bias
            # update meta info
            layer.out_channels = k
            nextlayer.in_features = k * n
        elif ptype == 'Conv-BN-FC':
            n = int(nextlayer.in_features / layer.out_channels)
            T = torch.kron(T.contiguous(), torch.eye(n))
            nextlayer.weight = nn.Parameter(nextlayer.weight @ T.T,
                    requires_grad=True)
            # don't do anything for bias
            # BN
            extralayer.weight = nn.Parameter(extralayer.weight[k_idx].clone(),
                    requires_grad=True)
            extralayer.bias = nn.Parameter(extralayer.bias[k_idx].clone(),
                    requires_grad=True)
            extralayer.register_buffer('running_mean', 
                    extralayer.running_mean[k_idx].clone())
            extralayer.register_buffer('running_var',
                    extralayer.running_var[k_idx].clone())
            # update meta info
            layer.out_channels = k
            nextlayer.in_features = k * n
            extralayer.num_features = k
        elif ptype == 'Conv-Conv':
            Wnext = nextlayer.weight.clone()
            Wnext = Wnext.permute(0,2,3,1)
            Wnext = torch.matmul(Wnext, T.T)
            Wnext = Wnext.permute(0,3,1,2) # batch x broadcast
            nextlayer.weight = nn.Parameter(Wnext, requires_grad=True)
            # don't do anything for bias
            # update meta info
            layer.out_channels = k
            nextlayer.in_channels = k
        elif ptype =='Conv-BN-Conv':
            
            assert(extralayer is not None)
            assert(type(extralayer) == nn.BatchNorm2d)
            # Conv
            
            Wnext = nextlayer.weight.clone()
            Wnext = Wnext.permute(0,2,3,1)
            Wnext = torch.matmul(Wnext, T.T) # batch x broadcast
            Wnext = Wnext.permute(0,3,1,2)
            nextlayer.weight = nn.Parameter(Wnext, requires_grad=True)
            # don't do anything for bias
            # BN
            extralayer.weight = nn.Parameter(extralayer.weight[k_idx].clone(),
                    requires_grad=True)
            extralayer.bias = nn.Parameter(extralayer.bias[k_idx].clone(),
                    requires_grad=True)
            extralayer.register_buffer('running_mean', 
                    extralayer.running_mean[k_idx].clone())
            extralayer.register_buffer('running_var',
                    extralayer.running_var[k_idx].clone())
            # update meta info
            layer.out_channels = k
            nextlayer.in_channels = k
            extralayer.num_features = k

        else:
            raise NotImplementedError

def pruneID(layerls, Zfnls, ptypels, trackls, 
        X, k, mode='frac', arch=None, skip=[], params=[]):
    '''
    layerls: list of tuples (layer, nextlayer) specifying which 
             old layers to prune
    prunelayerls: new prune layers (prunelayer, prunenextlayer)
    Zfnls: list of fn of how to get Z
    ptypels: {'FC-FC','Conv-FC','Conv-Conv'} specify ID propagation type
    X: data to create Z 
    '''
    print("layers must be in forward pass order")
    if mode == 'frac': assert(0 < k and k <= 1)
    assert(len(layerls) == len(Zfnls) == len(ptypels))

    input=X

    original=X
    layerks=[]
    Zs=[]
    svds=[]
    layerout=original
    layernum=0
    for (layer, nextlayer, extralayer), Zfn, ptype, track_id \
            in zip(layerls, Zfnls, ptypels, trackls):

        with torch.no_grad():
            if layernum<43:
                layerout = Zfn[0][layernum:](layerout)
        Z=layerout
        for i in range(1, len(Zfn)):
            with torch.no_grad():
                Z=Zfn[i](Z)
        Zs.append(Z.detach().numpy())
        layernum=len(Zfn[0])
        layerk , svd= getK(k, layer, track_id, mode=mode, A=Z, arch=arch, skip=skip)
        layerks.append(layerk)
        svds.append(svd)

    svds=np.array(svds)
    params=np.array(params)
    para=params[:-1]*.1+params[1:]*.1
    scores=(svds/para)
    print(scores)
    prune=np.argmin(scores)
    print(prune)
    (layer, nextlayer, extralayer)=layerls[prune]
    Zfn=Zfnls[prune]
    ptype=ptypels[prune]
    track_id=trackls[prune]
    layerk=layerks[prune]
    Z=Zs[prune] 
    pruneIDLayer(layer, nextlayer, Z, layerk, ptype, extralayer, ft=True)


            

def getK(k, layer, track_id, mode='frac', A=None, arch=None, skip=[]):
    '''
    calculates k for a layer
    mode: frac, direct, acc
    '''

    #print(track_id)
    # calculate old num neurons
    if type(layer) == nn.Conv2d:
        n = layer.out_channels
        assert(n == layer.weight.shape[0])
    elif type(layer) == nn.Linear:
        n = layer.out_features
        assert(n == layer.weight.shape[0])
    # k modes
    if mode == 'frac':
        newk = int(k * n)
        assert(newk <= n)
        return newk

    elif mode == 'fracSkipIter':
        assert(arch is not None)
        if arch =="ResNet56":
            if track_id in [16,20,38,54]:
                return n
            else:
                return getK(k, layer, track_id, 'frac', A, arch)
        elif arch == "VGG16":
            svd= ln.svd(A, compute_uv=False, full_matrices=False)
            k=getK(k, layer, track_id, 'frac', A, arch)
            return k, svd[k]/svd[0]

        else:
            raise NotImplementedError



def choosePruneMethod(args, model_full, X, k):
    model_full.eval()

    if args.pruner == "id":
        if args.arch == "AlexNet" or \
                args.arch == "ResNet56" or args.arch == "ResNet50" or \
                args.arch == "VGG16":
            assert(0 < k and k <= 1.0)
            summary_input = (3,32,32)
            ms, conv, lin=model_summary(model_full,summary_input= summary_input, input_res=32)
            params=conv+lin
            
            model_id = copy.deepcopy(model_full)
            model_id.eval()
            if args.pruner_args == "Zorig":
                pruneID(model_id.layers, model_full.Zfns, model_id.ptypes,
                        model_id.tracking,
                        X, k, mode=args.k_args, arch=args.arch, skip=args.skip, params=params)
            else:
                raise NotImplementedError


    return model_id


