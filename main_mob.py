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
import mobilenet_imgnet
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
    args.batch_size = 64 #1000
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

def calc_flops():
    '''
    calculate flops at various pruning fractions
    '''
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_full = nn_utils.load_model().to(device)
    ms, conv, lin = nn_utils.model_summary(model_full, dataset="imagenet",
                                            summary_input=(3,224,224), input_res=224)

    for _k in [0.95, 0.9, 0.85, 0.8, 0.75, 0.65, 0.6, 0.5, 0.25]:
        print(f"k={_k}")
        model_id = torch.load(f"./logs/id_k{_k}_pbs500/modelIDprune.pt", 
                                map_location=device)
        ms, conv, lin = nn_utils.model_summary(model_id, dataset="imagenet",
                                                summary_input=(3,224,224), input_res=224)
                        


    ##model_full = model_full.to(device)
    #prune_loader, test_loader = data_utils.imagenet_val_loader(args)
    #X_prune, _ = next(iter(prune_loader))
    #X_prune = X_prune.to(device)

    ##print(f"FLOPs of original VGG16 model")
    ##ms, conv, lin = nn_utils.model_summary(model_full, dataset="imagenet",
    ##                                        summary_input=(3,224,224), input_res=224)

    ##for k in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #for k in [0.6, 0.7, 0.8, 0.9]:
    #    args.k = k
    #    # prune
    #    model_id = id_utils.choosePruneMethod(args, model_full, X_prune, device)
    #    # calc flops
    #    print(f"FLOPs of pruned model {args.k}")
    #    ms, conv, lin = nn_utils.model_summary(model_id, summary_input=(3,224,224), input_res=224)

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
def id(args, checkdir="."):
    '''
    one-shot ID
    '''
    # seed in val_loader
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_full = nn_utils.load_model().to(device)
    prune_loader, test_loader = data_utils.imagenet_val_loader(args)
    X_prune, _ = next(iter(prune_loader))
    X_prune = X_prune.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # prune
    print(f"Pruning one-shot ID with X_prune {X_prune.shape}...")
    t0 = time.time()
    #model_id = id_utils.prune_ID(args, model_full, X_prune, device)
    model_id = mobilenet_imgnet.MobileNetPruned_ID(model_full, X_prune, k=args.k) 
    torch.save(model_id, f"{checkdir}/modelIDprune.pt")
    print(f"took {time.time() - t0} seconds")

    # dataparallel
    if torch.cuda.device_count() > 1:
        model_full = torch.nn.DataParallel(model_full)
        model_id = torch.nn.DataParallel(model_id)
    print("move models to dataparallel")

    del X_prune
    print("del X_prune")

    # evaluate
    # one-off for model_full eval
    #if (args.k == 0.25) and (args.prune_batch_size == 250):
    #    nn_utils.test(model_full, model_full, device, test_loader, criterion, 0)


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
    # seed in val_loader
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_full   = nn_utils.load_model()
    model_iditer = copy.deepcopy(model_full)
    #if torch.cuda.device_count() > 1:
    #    model_full = torch.nn.DataParallel(model_full)
    #    model_iditer = torch.nn.DataParallel(model_iditer)
    model_full   = model_full.to(device)
    model_iditer = model_iditer.to(device)
    prune_loader, test_loader = data_utils.imagenet_val_loader(args)
    X_prune, _   = next(iter(prune_loader))
    X_prune      = X_prune.to(device)
    criterion    = torch.nn.CrossEntropyLoss()
    #Rreduc_dict, Rpivot_dict, last_idx = None, None, np.inf

    # initial model summary
    ms_full, _, _ = nn_utils.model_summary(model_full, summary_input=summary_input, input_res=224)
    ms = ms_full
    cnt = 0
    while ms>ms_full * flops_reduc:
        print("")
        print(f"======= iteration {cnt} =======")
        t0 = time.time()
        model_iditer = mobilenet_imgnet.MobileNetPruned_IDiter(original=model_iditer, X_prune=X_prune, k=args.k)
        ms, _, _ = nn_utils.model_summary(model_iditer, summary_input=summary_input, input_res=224)
        print(f"took {time.time() - t0} seconds", flush=True)

        if cnt % args.log_interval == 0:
            # load pretrained model
            t0 = time.time()
            del model_full
            model_full = nn_utils.load_model()
            model_full = model_full.to(device)
            if torch.cuda.device_count() > 1:
                model_full = torch.nn.DataParallel(model_full)
            nn_utils.test(model_iditer, model_full, device, test_loader, criterion, 0)
            #nn_utils.test(model_iditer, model_full, device, prune_loader, criterion, 0)
            torch.save(model_iditer, f"{checkdir}/modelIDiter_cnt{cnt}.pt")
            print(f"took {time.time() - t0} seconds")
        cnt += 1
        
        # save model copy for Zorig
        del model_full
        model_full = copy.deepcopy(model_iditer)

    print("")
    print("======== Final Accuracy: ===========")
    del model_full
    model_full = nn_utils.load_model().to(device)
    nn_utils.test(model_iditer, model_full, device, test_loader, criterion, cnt)
    torch.save(model_iditer, f"{checkdir}/modelIDiter_final.pt")

