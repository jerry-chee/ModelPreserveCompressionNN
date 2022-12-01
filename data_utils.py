import os
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms as transforms
from typing import Iterator, List, Optional, Union
from operator import itemgetter
#from pytorch_lmdb_imagenet.folder2lmdb import ImageFolderLMDB
import folder2lmdb_v2

def dataset_split(args, test_set):
    '''
    separate test set into prune, test
    '''
    # consistent prune / test set based on random seed
    dataset_size = len(test_set)
    indices = list(range(dataset_size))
    split = int(args.prune_batch_size)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    test_indices, val_indices = indices[split:], indices[:split]

    # sampler
    # DataLoader(shuffle=True) mutually exclusive with SubsetRandomSampler
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    prune_loader = DataLoader(test_set, 
            batch_size=args.batch_size, # this needs to be smaller to load data for torchprune
            sampler=val_sampler,
            shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set,
            batch_size=args.batch_size,
            sampler=test_sampler,
            shuffle=False, pin_memory=True, num_workers=args.workers)

    return prune_loader, test_loader

def print_datastats(train_loader, prune_loader, test_loader): #, ft_loader):
    print("training/ft length: {}".format(len(train_loader.sampler)))
    print("pruning length: {}".format(len(prune_loader.sampler)))
    print("test length: {}".format(len(test_loader.sampler)))
    #print("finetune length: {}".format(len(ft_loader.sampler)))

def imagenet_train_loader(args):
    '''
    loads imagenet training data using lmdb
    '''
    traindir = "/share/desa/yl2967/imagenet/train.lmdb"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    train_dataset = folder2lmdb_v2.ImageFolderLMDB(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ]))
    if args.distributed: 
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True)
    
    return train_loader, train_sampler

def imagenet_val_loader(args):
    '''
    loads imagenet val data using lmdb, splits into prune/test set
    '''
    valdir = "/share/desa/yl2967/imagenet/val.lmdb"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    val_dataset = folder2lmdb_v2.ImageFolderLMDB(
        db_path=valdir,
        transform=
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    prune_loader, test_loader = dataset_split(args, val_dataset)
    #val_loader = torch.utils.data.DataLoader(
    #    val_dataset, batch_size=args.batch_size, shuffle=False,
    #    num_workers=args.workers, pin_memory=True)
    return prune_loader, test_loader


class DatasetFromSampler(Dataset):
    """
    Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """
    def __init__(self, sampler: torch.utils.data.sampler.Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """
        Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(torch.utils.data.DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))