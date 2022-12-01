
# import the required packages
import os
import copy
import time
import argparse
import pickle
import torch
import torchvision
import torchprune as tp
import vgg_imagenet
import mobilenet_imgnet
import data_utils
import nn_utils

def create_args():
    parser           = argparse.ArgumentParser(description='')
    args             = parser.parse_args(args=[])

    args.dataset     = "imagenet"
    args.batch_size  = 32 #1000
    args.prune_batch_size = 5000 # match iditer
    args.seed = 2 #match iditer

    # distibuted
    args.distributed       = False
    args.workers           = 4 #64

    return args

@torch.no_grad()
def lrank_prune(keep_ratio, prune_size, savename="vgg", net=None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initilaize network and wrap it into NetHandle class
    print("\n===========================")
    print("Load pretrained model, wrap into NetHandle.", flush=True)
    if net is None:
        net = vgg_imagenet.vgg16_imagenet(pretrained=True)
    #net = mobilenet_imgnet.load_model()
    net = tp.util.net.NetHandle(net)

    args = create_args()
    args.workers = 8
    args.prune_size = prune_size
    args.batch_size = 4
    prune_loader, test_loader = data_utils.imagenet_val_loader(args)

    # Prune filters
    print("\n===========================")
    print("Pruning filters with LRank.", flush=True)
    criterion = torch.nn.CrossEntropyLoss()
    # need to initialize
    net = net.cpu()
    net.forward(torch.ones(1,3,224,224))
    net_lrank = tp.LearnedRankNet(net, prune_loader, loss_handle=criterion)
    net_lrank = net_lrank.to(device)
    net_lrank.compress(keep_ratio=keep_ratio)

    print("\n===========================", flush=True)
    print("Saving LRank")
    with open(f"checkpoints/lrank/{savename}lrank_k{keep_ratio}_p{prune_size}.tp", 'wb') as f:
        #pickle.dump(net_lrank.compressed_net.torchnet.cpu(), f)
        pickle.dump(net_lrank.compressed_net.cpu(), f)


    # Measure flops
    print("\n===========================", flush=True)
    print("Measuring FLOPs.")
    net = net.to(device)
    nn_utils.model_summary(net, summary_input=(3,224,224), input_res=224)
    nn_utils.model_summary(net_lrank.compressed_net.torchnet.cuda(), summary_input=(3,224,224), input_res=224)

    #if torch.cuda.device_count() > 1:
    #    net = torch.nn.DataParallel(net)
    #    net_lrank.compressed_net.torchnet = torch.nn.DataParallel(net_lrank.compressed_net.torchnet)
    args.batch_size = 32

    # test
    print("\n===========================", flush=True)
    print("Evaluating ...")
    print("original model")
    nn_utils.test(net, net, device, test_loader, criterion, 0, calc_corr=False)
    t0 = time.time()
    nn_utils.test(net_lrank, net, device, test_loader, criterion, 0)
    print(f"took {time.time() - t0} seconds")