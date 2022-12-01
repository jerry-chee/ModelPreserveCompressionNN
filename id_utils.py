import copy
import torch
import torch.nn as nn
import numpy as np
import scipy.linalg
import nn_utils
import time

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
    # column-pivot QR
    R, P = scipy.linalg.qr((Z), mode='r', pivoting=True)
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

@torch.no_grad()
def pruneIDLayer(layer, nextlayer, Z, k, ptype='FC-FC', \
    extralayer=None, device=torch.device("cpu")):
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
    # select neurons and calculate interpolation
    # assign in place weight (and bias)
    k_idx, T = getID(k, Z, mode='kT')
    layer.weight = nn.Parameter(layer.weight[k_idx,:].clone().contiguous(), 
            requires_grad=True)
    if biasbool:
        layer.bias = nn.Parameter(layer.bias[k_idx].clone().contiguous(), 
                requires_grad=True)
    T = torch.Tensor(T).to(device)

    # propagation to next layer
    if ptype == 'FC-FC':
        nextlayer.weight = nn.Parameter((nextlayer.weight @ T.T).contiguous(),
                requires_grad=True)
        # don't do anything for bias
        # update meta info
        layer.out_features = k
        nextlayer.in_features = k
    elif ptype == 'FC-BN-FC':
        nextlayer.weight = nn.Parameter((nextlayer.weight @ T.T).contiguous(),
                requires_grad=True)
        # don't do anything for bias
        # BN
        extralayer.weight = nn.Parameter(extralayer.weight[k_idx].clone().contiguous(),
                requires_grad=True)
        extralayer.bias = nn.Parameter(extralayer.bias[k_idx].clone().contiguous(),
                requires_grad=True)
        extralayer.register_buffer('running_mean', 
                extralayer.running_mean[k_idx].clone().contiguous())
        extralayer.register_buffer('running_var',
                extralayer.running_var[k_idx].clone().contiguous())
        # update meta info
        layer.out_features = k
        nextlayer.in_features = k
        extralayer.num_features = k
    elif ptype == 'Conv-FC':
        n = int(nextlayer.in_features / layer.out_channels)
        T = torch.kron(T.contiguous(), torch.eye(n).to(device))
        nextlayer.weight = nn.Parameter((nextlayer.weight @ T.T).contiguous(),
                requires_grad=True)
        # don't do anything for bias
        # update meta info
        layer.out_channels = k
        nextlayer.in_features = k * n
    elif ptype == 'Conv-BN-FC':
        n = int(nextlayer.in_features / layer.out_channels)
        T = torch.kron(T.contiguous(), torch.eye(n).to(device))
        nextlayer.weight = nn.Parameter((nextlayer.weight @ T.T).contiguous(),
                requires_grad=True)
        # don't do anything for bias
        # BN
        extralayer.weight = nn.Parameter(extralayer.weight[k_idx].clone().contiguous(),
                requires_grad=True)
        extralayer.bias = nn.Parameter(extralayer.bias[k_idx].clone().contiguous(),
                requires_grad=True)
        extralayer.register_buffer('running_mean', 
                extralayer.running_mean[k_idx].clone().contiguous())
        extralayer.register_buffer('running_var',
                extralayer.running_var[k_idx].clone().contiguous())
        # update meta info
        layer.out_channels = k
        nextlayer.in_features = k * n
        extralayer.num_features = k
    elif ptype == 'Conv-Conv':
        Wnext = nextlayer.weight.clone()
        Wnext = Wnext.permute(0,2,3,1)
        Wnext = torch.matmul(Wnext, T.T)
        Wnext = Wnext.permute(0,3,1,2) # batch x broadcast
        nextlayer.weight = nn.Parameter(Wnext.contiguous(), requires_grad=True)
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
        nextlayer.weight = nn.Parameter(Wnext.contiguous(), requires_grad=True)
        # don't do anything for bias
        # BN
        extralayer.weight = nn.Parameter(extralayer.weight[k_idx].clone().contiguous(),
                requires_grad=True)
        extralayer.bias = nn.Parameter(extralayer.bias[k_idx].clone().contiguous(),
                requires_grad=True)
        extralayer.register_buffer('running_mean', 
                extralayer.running_mean[k_idx].clone().contiguous())
        extralayer.register_buffer('running_var',
                extralayer.running_var[k_idx].clone().contiguous())
        # update meta info
        layer.out_channels = k
        nextlayer.in_channels = k
        extralayer.num_features = k
    else:
        raise NotImplementedError

@torch.no_grad()
def tsqr(X, f, blocksize=25):
    '''
    TSQR
    '''
    out_ls = []
    for X_i in torch.split(X, blocksize, dim=0):
        out_i = f(X_i)
        _, R_i = torch.linalg.qr(out_i, mode='r')
        out_ls.append(R_i)
    out = torch.cat(out_ls, dim=0)
    return out

@torch.no_grad()
def prunevgg_ID_tsqr(args, layerls, Zfnls, ptypels, trackls, 
    X, k, mode='frac', arch=None, device=torch.device("cpu")):
    '''
    prune using TSQR approach
    '''
    assert mode == "frac" and (0 < k and k <= 1)
    assert(len(layerls) == len(Zfnls) == len(ptypels)) 
    cpu_device = torch.device("cpu")

    # init saving
    R_dict = {}
    for i in trackls:
        R_dict[i] = []

    # compute reduced Rs, TSQR
    for X_i in torch.split(X, args.forward_blocksize, dim=0):
        for (layer, nextlayer, extralayer), Zfn, ptype, track_id \
                in zip(layerls, Zfnls, ptypels, trackls):
            # define layer
            if track_id < 14:
                # before classifier
                if track_id == 1:
                    layer_fn = Zfn[0]
                else:
                    prev = len(Zfnls[track_id-2][0])
                    layer_fn = Zfn[0]
                    layer_fn = layer_fn[prev:]
                # layer forward
                X_i = layer_fn(X_i)
                # reshape
                Z_i = Zfn[1](X_i)
                # TSQR
                _, R_i = torch.linalg.qr(Z_i, mode='r')
                # save
                R_dict[track_id].append(R_i.to(cpu_device, non_blocking=True))
            elif track_id == 14:
                # boundary
                layer_fn = Zfn[1:]
                # layer forward
                X_i = layer_fn(X_i)
                # check if TSQR worth it
                if X.shape[0] > X.shape[1] + 1000:
                    # no reshape
                    # TSQR
                    _, R_i = torch.linalg.qr(X_i, mode='r')
                    # save
                    R_dict[track_id].append(R_i.to(cpu_device, non_blocking=True))
                else:
                    R_dict[track_id].append(X_i.to(cpu_device, non_blocking=True))
            else:
                # within classifier
                layer_fn = Zfn[1:]
                layer_fn = layer_fn
                layer_fn[1] = layer_fn[1][3:]
                # layer forward
                X_i = layer_fn(X_i)
                # check if TSQR worth it
                if X.shape[0] > X.shape[1] + 1000:
                    # no reshape
                    # TSQR
                    _, R_i = torch.linalg.qr(X_i, mode='r')
                    # save
                    R_dict[track_id].append(R_i.to(cpu_device, non_blocking=True))
                else:
                    R_dict[track_id].append(X_i.to(cpu_device, non_blocking=True))

    # aggregate
    for i in trackls:
        R_dict[i] = torch.cat(R_dict[i], dim=0)

    # ID
    for (layer, nextlayer, extralayer), Zfn, ptype, track_id, \
            in zip(layerls, Zfnls, ptypels, trackls):
        Z = R_dict[track_id].detach().numpy()
        layerk = getK(k, layer, track_id, mode=mode, A=Z)
        pruneIDLayer(layer, nextlayer, Z, layerk, ptype, extralayer, device)

@torch.no_grad()
def prunevgg_IDiter_tsqr(args, layerls, Zfnls, ptypels, trackls, 
    X, k, paramls=[], Rreduc_dict=None, Rpivot_dict=None, last_idx=np.inf,
    device=torch.device("cpu")):
    '''
    prune iditer one step using TSQR
    R_dict previous version of reduced Rs
    last_idx previous pruning index, only need to recompute after this
    '''
    mode = "iditer"
    assert (0 < k and k <= 1)
    assert(len(layerls) == len(Zfnls) == len(ptypels)) 
    assert ((Rreduc_dict is None) and (Rpivot_dict is None)) or ((Rreduc_dict is not None) and (Rpivot_dict is not None))
    if (Rpivot_dict is None): assert (last_idx is np.inf)
    if (last_idx is np.inf): assert (Rpivot_dict is None)
    first_pass = False
    if Rpivot_dict is None: first_pass = True
    cpu_device = torch.device("cpu")

    # reduced R saving
    if first_pass:
        Rreduc_dict = {}
        for i in trackls:
            Rreduc_dict[i] = []
    elif (not first_pass):
        for i in trackls:
            if i >= last_idx+1:
                Rreduc_dict[i] = [] 

    # compute reduced Rs, TSQR
    for X_i in torch.split(X, args.forward_blocksize, dim=0):
        for Zfn, track_id in zip(Zfnls, trackls):
            # no pre-saves, or after last_idx
            if (first_pass) or ((not first_pass) and (track_id >= last_idx+1)):
                # define layer
                if track_id < 14:
                    # before classifier
                    if track_id == 1:
                        layer_fn = Zfn[0]
                    else:
                        prev = len(Zfnls[track_id-2][0])
                        layer_fn = Zfn[0]
                        layer_fn = layer_fn[prev:]
                    # layer forward
                    X_i = layer_fn(X_i)
                    # reshape
                    Z_i = Zfn[1](X_i)
                    # TSQR
                    _, R_i = torch.linalg.qr(Z_i, mode='r')
                    # save
                    Rreduc_dict[track_id].append(R_i.to(cpu_device, non_blocking=True))
                elif track_id == 14:
                    # boundary
                    layer_fn = Zfn[1:]
                    # layer forward
                    X_i = layer_fn(X_i)
                    # check if TSQR worth it
                    if X.shape[0] > X.shape[1] + 1000:
                        # no reshape
                        # TSQR
                        _, R_i = torch.linalg.qr(X_i, mode='r')
                        # save
                        Rreduc_dict[track_id].append(R_i.to(cpu_device, non_blocking=True))
                    else:
                        Rreduc_dict[track_id].append(X_i.to(cpu_device, non_blocking=True))
                else:
                    # within classifier
                    layer_fn = Zfn[1:]
                    layer_fn = layer_fn
                    layer_fn[1] = layer_fn[1][3:]
                    # layer forward
                    X_i = layer_fn(X_i)
                    # check if TSQR worth it
                    if X.shape[0] > X.shape[1] + 1000:
                        # no reshape
                        # TSQR
                        _, R_i = torch.linalg.qr(X_i, mode='r')
                        # save
                        Rreduc_dict[track_id].append(R_i.to(cpu_device, non_blocking=True))
                    else:
                        Rreduc_dict[track_id].append(X_i.to(cpu_device, non_blocking=True))
            # pre-saves, but need to update X_i up to last_idx
            elif ((not first_pass) and (track_id == (last_idx+1) - 1)):
                # define layer
                if track_id < 14:
                    # before classifier
                    layer_fn = Zfn[0]
                    # layer forward
                    X_i = layer_fn(X_i)
                else :
                    # boundary
                    layer_fn = Zfn
                    # layer forward
                    X_i = layer_fn(X_i)

    # aggregate relevant, remove non-compute keys
    for i in Rreduc_dict.keys():
        if isinstance(Rreduc_dict[i], list):
            Rreduc_dict[i] = torch.cat(Rreduc_dict[i], dim=0)
        #if len(Rreduc_dict[i]) > 0:
        #    Rreduc_dict[i] = torch.cat(Rreduc_dict[i], dim=0)

    # compute FLOPs weighted ID score
    if first_pass:
        Rpivot_dict = {}
    layerk_ls, errls = [], []
    for (layer, _, _), Zfn, track_id, in zip(layerls, Zfnls, trackls):
        # no pre-saves or after last_idx
        if (first_pass) or ((not first_pass) and (track_id >= last_idx+1)):
            Z = Rreduc_dict[track_id].detach().numpy()
            R_i, _ = scipy.linalg.qr(Z, mode='r', pivoting=True)
            Rpivot_dict[track_id] = R_i
            layerk_i, err_i = getK(k, layer, track_id, mode=mode, R=R_i)
            layerk_ls.append(layerk_i)
            errls.append(err_i)
        else:
            R_i = Rpivot_dict[track_id]
            layerk_i, err_i = getK(k, layer, track_id, mode=mode, R=R_i)
            layerk_ls.append(layerk_i)
            errls.append(err_i)

    # select layer to prune    
    errls   = np.array(errls)
    paramls = np.array(paramls)
    paramls = (paramls[:-1] * 0.1) + (paramls[1:] * 0.1)
    scores  = errls / paramls
    print(f"errs: {errls}")
    print(f"params: {paramls}")
    print(f"scores: {scores}")
    prune_id = np.argmin(scores)
    (layer, nextlayer, extralayer) = layerls[prune_id]
    print(f"selected layer {prune_id} {layer}")
    Zfn = Zfnls[prune_id]
    ptype = ptypels[prune_id]
    track_id = trackls[prune_id]
    layerk = layerk_ls[prune_id]
    Z = Rreduc_dict[track_id] # track_id is 1-index
    pruneIDLayer(layer, nextlayer, Z, layerk, ptype, extralayer, device=device)
    return Rreduc_dict, Rpivot_dict, prune_id

@torch.no_grad()
def getK(k, layer, track_id, mode='frac', R=None):
    '''
    calculates k for a layer
    mode: frac, direct, acc
    '''
    #print(mode)
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
    elif mode == "iditer":
        assert(R is not None)
        k = getK(k, layer, track_id, 'frac')
        err = np.abs(R[k,k] / R[0,0])
        return k, err
    else:
        raise NotImplementedError

@torch.no_grad()
def prune_ID(args, model_full, X, device):
    assert args.pruner == "id"
    assert args.k_args == "frac"
    assert(0 < args.k and args.k <= 1.0)
    model_full.eval()
    model_id = copy.deepcopy(model_full)
    model_id.to(device)
    model_id.eval()

    # defaulit Zorig
    if args.arch == "VGG16":
        prunevgg_ID_tsqr(args, 
            model_id.layers, model_full.Zfns, 
            model_id.ptypes, model_id.tracking, 
            #model_id.module.layers, model_full.module.Zfns, 
            #model_id.module.ptypes, model_id.module.tracking, 
            X, args.k, device=device)
        return model_id
    else:
        raise NotImplementedError

@torch.no_grad()
def prune_IDiter(args, model_full, model_iditer, X, Rreduc_dict, Rpivot_dict, last_idx, device):
    assert args.pruner == "id"
    assert args.k_args == "iditer"
    assert(0 < args.k and args.k <= 1.0)
    assert isinstance(model_full, torch.nn.DataParallel) and isinstance(model_iditer, torch.nn.DataParallel)
    model_full.eval()
    model_iditer.eval()
    summary_input = (3,224,224)
    ms, conv, lin = nn_utils.model_summary(model_iditer, summary_input=summary_input, input_res=224)
    if torch.cuda.device_count() > 1:
        ms, conv, lin = nn_utils.model_summary(model_iditer.module, summary_input=summary_input, input_res=224)
    paramls = conv + lin

    # defaulit Zorig
    if args.arch == "VGG16":
        assert(0 < args.k and args.k <= 1.0)
        # DataParallel optional
        if torch.cuda.device_count() > 1:
            Rreduc_dict, Rpivot_dict, last_idx = \
            prunevgg_IDiter_tsqr(args,
                model_iditer.module.layers, model_full.module.Zfns,
                model_iditer.module.ptypes, model_iditer.module.tracking,
                X, args.k, 
                paramls=paramls, Rreduc_dict=Rreduc_dict, Rpivot_dict=Rpivot_dict, last_idx=last_idx,
                device=device)
        else:
            Rreduc_dict, Rpivot_dict, last_idx = \
            prunevgg_IDiter_tsqr(args,
                model_iditer.layers, model_full.Zfns,
                model_iditer.ptypes, model_iditer.tracking,
                X, args.k, 
                paramls=paramls, Rreduc_dict=Rreduc_dict, Rpivot_dict=Rpivot_dict, last_idx=last_idx,
                device=device)
        return Rreduc_dict, Rpivot_dict, last_idx
    else:
        raise NotImplementedError


