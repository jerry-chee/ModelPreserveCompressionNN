import argparse
import time
import os
import copy
import torch
import numpy as np
import nn_utils
#from nn_utils import model_summary, test, load_model
import data_utils
import id_utils
#import nn_utils
#from nn_utils import train, test, load_model
#from fc import FC2, FC1
#from id_utils import pruneID, compare_prune, getID
#import data_utils
#from plt_utils import plt_IDerr
#import torch.nn.utils.prune as prune
#import torch
#import torchvision
#import torchvision.transforms as transforms
#from train import model_summary
#import matplotlib.pyplot as plt
#import numpy as np
#import importlib
#import id_utils6
summary_input = (3,224,224)

def create_args():
    parser           = argparse.ArgumentParser(description='Foolbox Decision Boundary Dataset Test')
    args             = parser.parse_args(args=[])

    args.arch        = "VGG16"
    args.dataset     = "imagenet"
    args.load_fname  = None 
    #args.lr          = 1e-3 #1e-3 #0.01
    #args.lr_milestones  = [20,40,60,80]#[60, 120, 160] #[80, 120]
    #args.gamma       = 0.5 #0.1 #0.2
    #args.momentum    = 0.9
    #args.weight_decay= 5e-4 #1e-4
    #args.log_interval= 1
    #args.verbose     = True
    args.k           = 0.70
    args.pruner      = "id" 
    #args.saveR       = False
    #args.epochs      = 20 #160
    args.batch_size = 128 #1000
    args.prune_batch_size = 1000
    #args.pruner_args = "Zorig" # default
    args.k_args      = "frac" #"fracSkipMeg" #"frac"
    #args.skip        = []

    # distibuted
    args.distributed       = False
    args.workers           = 4 #64
    args.forward_blocksize = 25
    #args.gausproj_blocksize = 2500

    return args

class Fake_Dataset(torch.utils.data.Dataset):
    '''
    DataLoader for decision boundary dataset
    '''
    def __init__(self, data, target):
        self.data   = data
        self.target = target
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def pretrain(args, seed=2):
    '''
    measure pretrained model accuracy
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_full = nn_utils.load_model(args)
    if torch.cuda.device_count() > 1:
        model_full = torch.nn.DataParallel(model_full)
    model_full = model_full.to(device)
    prune_loader, test_loader = data_utils.imagenet_val_loader(args)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Evaluating pretrained model with prune split size {args.prune_batch_size}...")
    t0 = time.time()
    nn_utils.test(model_full, model_full, device, test_loader, criterion, 0, calc_corr=False)
    print(f"took {time.time() - t0} seconds")

def calc_flops(args, seed=2):
    '''
    calculate flops at various pruning fractions
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_full = nn_utils.load_model(args)
    model_full = model_full.to(device)
    prune_loader, test_loader = data_utils.imagenet_val_loader(args)
    X_prune, _ = next(iter(prune_loader))
    X_prune = X_prune.to(device)

    #print(f"FLOPs of original VGG16 model")
    #ms, conv, lin = nn_utils.model_summary(model_full, dataset="imagenet",
    #                                        summary_input=(3,224,224), input_res=224)

    #for k in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for k in [0.6, 0.7, 0.8, 0.9]:
        args.k = k
        # prune
        model_id = id_utils.choosePruneMethod(args, model_full, X_prune, device)
        # calc flops
        print(f"FLOPs of pruned model {args.k}")
        ms, conv, lin = nn_utils.model_summary(model_id, summary_input=(3,224,224), input_res=224)

def compare_models(args, seed=2):
    '''
    compare 2 models correlation
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    prune_loader, test_loader = data_utils.imagenet_val_loader(args)
    criterion = torch.nn.CrossEntropyLoss()

    #model_id1k = torch.load("./logs/id_1kprune/modelIDprune.pt")
    #model_id5k = torch.load("./logs/id_5kprune/modelIDprune.pt")
    model_id7_5k = torch.load("./logs/id_7.5kprune/modelIDprune.pt")
    model_id10k = torch.load("./logs/id_10kprune/modelIDprune.pt")

    nn_utils.test(model_id7_5k, model_id10k, device, test_loader, criterion, 0)

@torch.no_grad()
def id(args, checkdir=".", seed=2):
    '''
    one-shot ID
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_full = nn_utils.load_model(args).to(device)
    prune_loader, test_loader = data_utils.imagenet_val_loader(args)
    X_prune, _ = next(iter(prune_loader))
    X_prune = X_prune.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # prune
    print(f"Pruning one-shot ID with X_prune {X_prune.shape}...")
    t0 = time.time()
    model_id = id_utils.prune_ID(args, model_full, X_prune, device)
    torch.save(model_id, f"{checkdir}/modelIDprune.pt")
    print(f"took {time.time() - t0} seconds")

    # evaluate
    print("Evaluating ...")
    t0 = time.time()
    nn_utils.test(model_id, model_full, device, test_loader, criterion, 0)
    print(f"took {time.time() - t0} seconds")

@torch.no_grad()
def iditer(args, flops_reduc=0.5, checkdir=".", seed=2, crashstart=False):
    '''
    iterative ID algorithm
    flops_reduc controls final flops reduction
    args.k sets prune stepsize
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_full   = nn_utils.load_model(args)
    model_iditer = copy.deepcopy(model_full)
    if torch.cuda.device_count() > 1:
        model_full = torch.nn.DataParallel(model_full)
        model_iditer = torch.nn.DataParallel(model_iditer)
    model_full   = model_full.to(device)
    model_iditer = model_iditer.to(device)
    prune_loader, test_loader = data_utils.imagenet_val_loader(args)
    X_prune, _   = next(iter(prune_loader))
    X_prune      = X_prune.to(device)
    criterion    = torch.nn.CrossEntropyLoss()
    Rreduc_dict, Rpivot_dict, last_idx = None, None, np.inf

    # initial model summary
    ms_full, _, _ = nn_utils.model_summary(model_full, summary_input=summary_input, input_res=224)
    ms = ms_full
    cnt = 0
    while ms>ms_full * flops_reduc:
        print("")
        print(f"======= iteration {cnt} =======")
        t0 = time.time()
        Rreduc_dict, Rpivot_dict, last_idx = \
        id_utils.prune_IDiter(args, model_full, model_iditer, X_prune,
            Rreduc_dict, Rpivot_dict, last_idx, device)
        ms, _, _ = nn_utils.model_summary(model_iditer, summary_input=summary_input, input_res=224)
        print(f"took {time.time() - t0} seconds", flush=True)

        if cnt % args.log_interval == 0:
            # load pretrained model
            t0 = time.time()
            del model_full
            model_full = nn_utils.load_model(args)
            if torch.cuda.device_count() > 1:
                model_full = torch.nn.DataParallel(model_full)
            model_full = model_full.to(device)
            nn_utils.test(model_iditer, model_full, device, test_loader, criterion, 0)
            torch.save(model_iditer, f"{checkdir}/modelIDiter_cnt{cnt}.pt")
            print(f"took {time.time() - t0} seconds")
        cnt += 1
        
        # save model copy for Zorig
        del model_full
        model_full = copy.deepcopy(model_iditer)

    print("")
    print("======== Final Accuracy: ===========")
    model_full = nn_utils.load_model(args).to(device)
    nn_utils.test(model_iditer, model_full, device, test_loader, criterion, cnt)
    torch.save(model_iditer, f"{checkdir}/modelIDiter_final.pt")


if __name__ == "__main__":
    args = create_args()

    #_checkdir = "./logs/iditer_5kprune_0.1step"
    #if not os.path.isdir(_checkdir):
    #    os.mkdir(_checkdir)
    #_seed = 2
    #args.prune_batch_size = 5000
    #args.k = 0.9
    #args.log_interval = 5
    #args.k_args = "iditer"
    #print(f"IDiter with stepsize {args.k}, prune_set {args.prune_batch_size}, seed {_seed}")
    #iditer(args, checkdir=_checkdir, seed=_seed)

    _checkdir = "./logs/iditer_5kprune_0.02step"
    if not os.path.isdir(_checkdir):
        os.mkdir(_checkdir)
    _seed = 2
    args.prune_batch_size = 5000
    args.k = 0.98
    args.log_interval = 10
    args.k_args = "iditer"
    print(f"IDiter with stepsize {args.k}, prune_set {args.prune_batch_size}, seed {_seed}")
    iditer(args, checkdir=_checkdir, seed=_seed)

    ## testing prune set size sensitivity

    #_checkdir = "./logs/id_1kprune"
    #if not os.path.isdir(_checkdir):
    #    os.mkdir(_checkdir)
    #print(f"====================================")
    #print(f"ID at frac {args.k}, prune_set {args.prune_batch_size}")
    #id(args, _checkdir, seed=2)

    #_checkdir = "./logs/id_2.5kprune"
    #if not os.path.isdir(_checkdir):
    #    os.mkdir(_checkdir)
    #args.prune_batch_size = 2500
    #print(f"====================================")
    #print(f"ID at frac {args.k}, prune_set {args.prune_batch_size}")
    #id(args, _checkdir, seed=2)

    #_checkdir = "./logs/id_5kprune"
    #if not os.path.isdir(_checkdir):
    #    os.mkdir(_checkdir)
    #args.prune_batch_size = 5000
    #print(f"====================================")
    #print(f"ID at frac {args.k}, prune_set {args.prune_batch_size}")
    #id(args, _checkdir, seed=2)

    #_checkdir = "./logs/id_7.5kprune"
    #if not os.path.isdir(_checkdir):
    #    os.mkdir(_checkdir)
    #args.prune_batch_size = 7500
    #print(f"====================================")
    #print(f"ID at frac {args.k}, prune_set {args.prune_batch_size}")
    #id(args, _checkdir, seed=2)

    #_checkdir = "./logs/id_10kprune"
    #if not os.path.isdir(_checkdir):
    #    os.mkdir(_checkdir)
    #args.prune_batch_size = 10000
    #print(f"====================================")
    #print(f"ID at frac {args.k}, prune_set {args.prune_batch_size}")
    #id(args, _checkdir, seed=2)

    #_checkdir = "./logs/id_15kprune"
    #if not os.path.isdir(_checkdir):
    #    os.mkdir(_checkdir)
    #args.prune_batch_size = 15000
    #print(f"====================================")
    #print(f"ID at frac {args.k}, prune_set {args.prune_batch_size}")
    #id(args, _checkdir, seed=2)

    #_checkdir = "./logs/id_20kprune"
    #if not os.path.isdir(_checkdir):
    #    os.mkdir(_checkdir)
    #args.prune_batch_size = 20000
    #print(f"====================================")
    #print(f"ID at frac {args.k}, prune_set {args.prune_batch_size}")
    #id(args, _checkdir, seed=2)

    #_checkdir = "./logs/id_25kprune"
    #if not os.path.isdir(_checkdir):
    #    os.mkdir(_checkdir)
    #args.prune_batch_size = 25000
    #print(f"====================================")
    #print(f"ID at frac {args.k}, prune_set {args.prune_batch_size}")
    #id(args, _checkdir, seed=2)
    
    ############################################################### 
    ## prune 1k at different fracs 
    #_checkdir = "./logs/id_1kprune_0.8frac"
    #if not os.path.isdir(_checkdir):
    #    os.mkdir(_checkdir)
    #args.k = 0.8
    #print(f"====================================")
    #print(f"ID at frac {args.k}, prune_set {args.prune_batch_size}")
    #id(args, _checkdir, seed=2)

    #_checkdir = "./logs/id_1kprune_0.9frac"
    #if not os.path.isdir(_checkdir):
    #    os.mkdir(_checkdir)
    #args.k = 0.9
    #print(f"====================================")
    #print(f"ID at frac {args.k}, prune_set {args.prune_batch_size}")
    #id(args, _checkdir, seed=2)



    
    ############################################################### 
    ## check pretrained model accuracy matches
    #args = create_args()
    #args.prune_batch_size = 5000
    #_seed=2
    #print(f"seed: {_seed}")
    #pretrain(args, seed=_seed)

    ############################################################### 
    ## calc FLOPs for model at various prune levels
    #args  = create_args()
    #_seed = 2
    #print(f"calc FLOPs for model at various prune levels")
    #calc_flops(args, seed=_seed)
