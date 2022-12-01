# from https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/imagenet/thinet/models/thinetconv.py
import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

import nn_utils


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}

out_shapes = [
    [64, 224, 224]
]

class transformZconv(nn.Module):
    def __init__(self):
        super(transformZconv, self).__init__()
    
    def forward(self, x):
        return nn_utils.transformZconv(x)

class convFCview(nn.Module):
    def __init__(self):
        super(convFCview, self).__init__()

    def forward(self, x):
        #return x.view(x.size(0), -1)
        return x.reshape(x.size(0), -1)

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, 
            cfg_classifier=None):
        super(VGG, self).__init__()
        self.features = features
        if cfg_classifier is None:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(cfg_classifier[0] * 7 * 7, cfg_classifier[1]),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(cfg_classifier[1], cfg_classifier[1]),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(cfg_classifier[1], num_classes),
            )

        if init_weights:
            self._initialize_weights()

        # for ID pruning
        self.transformZconv = transformZconv()
        self.convFCview = convFCview()
        self.layers = []
        self.Zfns   = []
        self.ptypes = []
        self.tracking = list(range(1,16))
        self._construct_layers()
        self._construct_Zfns()

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')#, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()  
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # for ID pruning
    def _construct_layers(self):
        def seq_helper(seq):
            for i, layer in enumerate(seq):
                if type(layer) == nn.Conv2d and len(seq) - i > 3:
                    if type(seq[i+2]) == nn.Conv2d:
                        self.layers.append((layer, seq[i+2], None))
                        self.ptypes.append('Conv-Conv')
                    elif type(seq[i+2]) == nn.MaxPool2d:
                        self.layers.append((layer, seq[i+3], None))
                        self.ptypes.append('Conv-Conv')
                    else:
                        raise NotImplementedError
                elif type(layer) == nn.Conv2d: # last layer
                    self.layers.append((layer, self.classifier[0], None))
                    self.ptypes.append('Conv-FC')


        # features 
        seq_helper(self.features)
        # classifier
        self.layers.append((self.classifier[0], 
            self.classifier[3], None))
        self.layers.append((self.classifier[3], 
            self.classifier[6], None))
        self.ptypes.append('FC-FC')
        self.ptypes.append('FC-FC')

    def _construct_Zfns(self):
        def seq_helper(seq):
            for i, layer in enumerate(seq):
                if type(layer) == nn.Conv2d and len(seq) - i > 3:
                    if type(seq[i+2]) == nn.Conv2d:
                        self.Zfns.append( nn.Sequential(\
                                self.features[0:i+2],\
                                self.transformZconv) )
                    elif type(seq[i+2]) == nn.MaxPool2d:
                        self.Zfns.append( nn.Sequential(\
                                self.features[0:i+3],\
                                self.transformZconv) )
                    else:
                        raise NotImplementedError
                elif type(layer) == nn.Conv2d: # last layer
                    self.Zfns.append( nn.Sequential(\
                            self.features,\
                            #self.convFCview) )
                            self.transformZconv) )

        # features
        seq_helper(self.features)
        #classifier
        self.Zfns.append( nn.Sequential(\
                self.features,
                self.convFCview,
                self.classifier[0:3] #linear,relu,dropout
                ))
        self.Zfns.append( nn.Sequential(\
                self.features,
                self.convFCview,
                self.classifier[0:6] #linear,relu,dropout,linear,relu,dropout
                ))

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg_official = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
cfg_prune = {
        0.25: [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'],
        0.50: [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
        0.75: [48, 48, 'M', 96, 96, 'M', 192, 192, 192, 'M', 384, 384, 384, 'M', 384, 384, 384, 'M']
        }
cfg_classifier = {
        0.00: [512,4096],
        0.25: [128,1024],
        0.50: [256,2048],
        0.75: [384,3072],
        }

def vgg16_imagenet(pretrained=False, #prune_reload=False, prune_k=None,
        **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    # TODO: extra args may not need
    #if prune_reload:
    #    assert prune_k is not None
    #    print("setting prune state to config, ", prune_k)
    #    cfg = cfg_prune[prune_k]
    #    cfg_class = cfg_classifier[prune_k]
    #else:
    #    cfg = cfg_official
    #    cfg_class = None
    cfg = cfg_official
    cfg_class = None
    model = VGG(make_layers(cfg, False), cfg_classifier=cfg_class, 
            **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model
