import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
import torchvision.transforms as transforms
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import torchvision
import torchvision.transforms as transforms


def dataset_split(args, train_set, test_set, shuffle=False):
    # separate sets for train, finetune, prune
    # first prune
    dataset_size = len(test_set)
    indices = list(range(dataset_size))
    split = int(args.prune_batch_size)
    np.random.shuffle(indices)
    test_indices, val_indices = indices[split:], indices[:split]

    # sampler
    # DataLoader(shuffle=True) mutually exclusive with SubsetRandomSampler
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    if train_set is not None:
        train_loader = DataLoader(train_set, 
                batch_size=args.batch_size,
                shuffle=True)
    else:
        train_loader = None
    prune_loader = DataLoader(test_set, 
            batch_size=args.prune_batch_size,
            sampler=val_sampler)
    test_loader = DataLoader(test_set,
            batch_size=args.test_batch_size, shuffle=False)
            #sampler=test_sampler)

    return train_loader, prune_loader, test_loader

def print_datastats(train_loader, prune_loader, test_loader): #, ft_loader):
    print("training/ft length: {}".format(len(train_loader.sampler)))
    print("pruning length: {}".format(len(prune_loader.sampler)))
    print("test length: {}".format(len(test_loader.sampler)))
    #print("finetune length: {}".format(len(ft_loader.sampler)))


def imagenet_loader(args):
    traindir = '~/data/imagenet2012/train'
    valdir = '~/data/imagenet2012/val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_set = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_set = torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = \
                torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if args.use_valid:
        _, prune_loader, val_loader = dataset_split(args, None, val_set)
    else:
        prune_loader = DataLoader(train_set, batch_size=args.prune_batch_size, 
                shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, 
                shuffle=False, num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, prune_loader, train_sampler
    

def cifar10_loader(args):
    #print("using very simple data preprocessing")
    #transform = transforms.Compose(
    #    [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    train_set = torchvision.datasets.CIFAR10(root='../data', 
            train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='../data', 
            train=False, download=True, transform=transform_test)

    if args.use_valid:
        train_loader, prune_loader, test_loader = \
                dataset_split(args, train_set, test_set)
    else:
        print("======================================================")
        print("NOTICE: Prune set from non held-out TRAIN but still pulling split out of TEST to maintain random seed accuracies...")
        print("======================================================")
        train_loader, _, test_loader = \
                dataset_split(args, train_set, test_set)
        #train_loader = DataLoader(train_set, batch_size=args.batch_size,
        #        shuffle=True)
        prune_loader = DataLoader(train_set, batch_size=args.prune_batch_size, 
                shuffle=True)
        #test_loader = torch.utils.data.DataLoader(test_set, 
        #        batch_size=args.test_batch_size, shuffle=False)

    print_datastats(train_loader, prune_loader, test_loader)
    return train_loader, test_loader, prune_loader

def fashionmnist_loader(args):
    train_set = torchvision.datasets.FashionMNIST(
                    root = '../data/FashionMNIST',
                    train = True,
                    download = True,target_transform=lambda y: torch.randint(0, 10, (1,)).item(),
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        
                        transforms.Normalize(mean=[0.2868], std=[0.3524]),
                        transforms.Lambda(lambda x: torch.flatten(x))
                        
                ])
    )
    test_set = torchvision.datasets.FashionMNIST(
                    root = '../data/FashionMNIST',
                    train = False,
                    download = True,
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.2868], std=[0.3524]),
                        transforms.Lambda(lambda x: torch.flatten(x)),
                        
                ])
    )
    print(train_set)
    if args.use_valid:
        train_loader, prune_loader, test_loader = dataset_split(args, train_set, test_set, shuffle=args.shuffle)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size,shuffle=args.shuffle)
        prune_loader = DataLoader(train_set, 
                            batch_size=args.prune_batch_size,shuffle=args.shuffle)

    test_loader  = DataLoader(test_set, batch_size=args.test_batch_size)

    return train_loader, test_loader, prune_loader


def patches_loader(args, p=2, r=0, s=1000, d=3):
    # patches dataset
    X_train, y_train, X_test, y_test = patches(p,r,s,d)

    # add column for bias
    vec1_train = np.ones((X_train.shape[0],1))
    vec1_test  = np.ones((X_test.shape[0] ,1))
    X_train    = np.concatenate((X_train, vec1_train), axis=1)
    X_test     = np.concatenate((X_test , vec1_test) , axis=1)

    y_train, y_test = y_train[:, None], y_test[:, None]
    patches_train = FromNumpyDataset(X_train, y_train)
    patches_test  = FromNumpyDataset(X_test , y_test)
    ptrain_loader = DataLoader(patches_train, batch_size=args.batch_size)
    ptest_loader  = DataLoader(patches_test ,
                                batch_size=args.test_batch_size)

    return ptrain_loader, ptest_loader, (X_train, y_train, X_test, y_test)


class FromNumpyDataset(Dataset):
    """Convert numpy arrays to torch dataloader"""

    def __init__(self, X, Y):
        """
        X is (n,d) 
        """
        if X.shape[0] != Y.shape[0]:
            warnings.warn("FromNumpyDataset(X and Y don't match in number samples)")
        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.X[idx], self.Y[idx]


def patches(p=2,r=0, s=1000,d=3):
    #note this is done the lazy way so using much 
    #more than d=10 breaks down.  
    if (d > 10): warnings.warn("d>10 breaks down patches()")

    points=np.random.random(size=(s*d**3,d))-.5
    points=points[(np.sum(points**2, axis=1)**.5)<.5]
    points=(points/(np.sum(points**2, axis=1)**.5)[:,None])[:s]

    y_train=np.zeros(s)
    X_train=points

    points=np.random.random(size=(s*d**3,d))-.5
    points=points[(np.sum(points**2, axis=1)**.5)<.5]
    points=(points/(np.sum(points**2, axis=1)**.5)[:,None])[:s]

    y_test=np.zeros(s)
    X_test=points

    patches=np.random.random((p*d**3*2,d))-.5
    patches=patches[(np.sum(patches**2, axis=1)**.5)<.5]
    patches=(patches/(np.sum(patches**2, axis=1)**.5)[:,None])[:p*2]

    additions=(X_train@patches.T)>r

    for i in range(p):
        y_train[additions[:, i]]+=1
    additions=(X_test@patches.T)>r

    for i in range(p):
        y_test[additions[:, i]]+=1 
    additions=(X_train@patches.T)>r

    for i in range(p, 2*p):
        y_train[additions[:, i]]-=1
    additions=(X_test@patches.T)>r

    for i in range(p, 2*p):
        y_test[additions[:, i]]-=1 

    return X_train, y_train, X_test, y_test

def threeVsFour():
    digits = sklearn.datasets.load_digits()
    n_samples = len(digits.images)

    images_and_labels = list(zip(digits.images, digits.target))
    data = digits.images.reshape((n_samples, -1))
    X_train, X_test, y_train, y_test = train_test_split( \
            data, digits.target, test_size=0.5, shuffle=False)
    inds=np.argsort(y_train)
    y_train=y_train[inds]
    X_train=X_train[inds]

    threes=np.min(np.where(y_train==3))
    fours=np.max(np.where(y_train==4))

    y_train=y_train[threes:fours+1]
    X_train=X_train[threes:fours+1]
    y_train[np.where(y_train==3)]=-1
    y_train[np.where(y_train==4)]=1

    inds=np.argsort(y_test)
    y_test=y_test[inds]
    X_test=X_test[inds]

    threes=np.min(np.where(y_test==3))
    fours=np.max(np.where(y_test==4))

    y_test=y_test[threes:fours+1]
    X_test=X_test[threes:fours+1]
    y_test[np.where(y_test==3)]=-1
    y_test[np.where(y_test==4)]=1
    mean=np.mean(X_train, axis=0)

    X_train=(X_train-mean[None, :])
    X_test=(X_test-mean[None, :])
    norms=np.sum(X_train**2, axis=1)**.5
    X_train=X_train/norms[:, None]
    norms=np.sum(X_test**2, axis=1)**.5
    X_test=X_test/norms[:, None]

    return X_train, y_train, X_test, y_test

def dataset_split_old(args, train_set, test_set):
    # separate sets for train, finetune, prune
    # first prune
    dataset_size = len(train_set)
    indices = list(range(dataset_size))
    split = int(args.prune_batch_size)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # sep set for train and finetune
    new_size = len(train_indices)
    split = int(args.ft_proportion * new_size) #proportion ft set
    np.random.shuffle(train_indices)
    ft_indices = train_indices[:split]
    train_indices = train_indices[split:]
    # sampler
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    ft_sampler = SubsetRandomSampler(ft_indices)

    train_loader = DataLoader(train_set, 
            batch_size=args.batch_size,
            sampler=train_sampler)
    prune_loader = DataLoader(train_set, 
            batch_size=args.prune_batch_size,
            sampler=val_sampler)
    ft_loader = DataLoader(train_set,
            batch_size=args.batch_size,
            sampler=ft_sampler)

    print_datastats(train_loader, test_set, prune_loader, ft_loader)
    if args.fine_tune:
        return ft_loader, prune_loader
    else:
        return train_loader, prune_loader

