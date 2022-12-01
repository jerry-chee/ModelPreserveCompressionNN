import os
import os.path as osp
from PIL import Image
import six
import lmdb
import pickle
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as F
import math
import torch

def loads_data(buf):
    """
    Args:
        buf: the output of dumps.
    """
    return pickle.loads(buf)

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.transform = transform
        self.target_transform = target_transform

        # https://github.com/chainer/chainermn/issues/129
        # Delay loading LMDB data until after initialization to avoid "can't pickle Environment Object error"
        self.env = None

        # Workaround to have length from the start for ImageNet since we don't have LMDB at initialization time
        if 'train' in self.db_path:
            self.length = 1281167
        elif 'val' in self.db_path:
            self.length = 50000
        else:
            raise NotImplementedError
    
    def _init_db(self):
        self.env = lmdb.open(self.db_path, subdir=osp.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()

        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self._class_._name_ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_data(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)


# comment haven't used this, may need to switch _keys_ -> __keys__, etc.
def folder2lmdb(spath, dpath, name="train", write_frequency=5000):
    directory = osp.expanduser(osp.join(spath, name))
    # directory = osp.expanduser(spath)
    print("Loading dataset from %s" % directory)
    dataset = ImageFolder(directory, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        image, label = data[0]

        txn.put(u'{}'.format(idx).encode('ascii'), dumps_data((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'_keys_', dumps_data(keys))
        txn.put(b'_len_', dumps_data(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


#if __name__ == "__main__":
#    # generate lmdb
#    # folder2lmdb("/data/imagenet/", "/data/users/v-yuchenglu/imagenet/", name="train")
#    folder2lmdb("/data/users/v-yuchenglu/imagenet2012/val/", "/data/users/v-yuchenglu/imagenet/", name="val")