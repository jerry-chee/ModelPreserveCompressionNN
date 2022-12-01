import copy
import shutil
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from id_utils import getID
from fc import FC1
from alexnet import AlexNet
from resnet_cifar10 import resnet_cifar
from vgg_cifar10 import vgg_cifar
#from vgg_imagenet import vgg16_imagenet
#from resnet_imagenet import resnet50_imagenet



def convShape(conv,H,W):
    '''
    calculates output shape for nn.Conv2d
    (Hin,Win) -> (Hout,Wout)
    '''
    Hout = (H + 2*conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0]-1) -1)/conv.stride[0] + 1
    Wout = (W + 2*conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1]-1) -1)/conv.stride[1] + 1
    return int(np.floor(Hout)), int(np.floor(Wout))


def poolShape(pool,H,W):
    '''
    calculates output shape for nn.MaxPool2d
    (Hin,Win) -> (Hout,Wout)
    '''
    def splitVec(v):
        if isinstance(v, list):
            return v[0], v[1]
        else:
            return v, v

    p0, p1 = splitVec(pool.padding)
    d0, d1 = splitVec(pool.dilation)
    k0, k1 = splitVec(pool.kernel_size)
    s0, s1 = splitVec(pool.stride)

    Hout = (H + 2*p0 - d0 * (k0-1) - 1)/s0 + 1
    Wout = (W + 2*p1 - d1 * (k1-1) -1)/s1 + 1
    return int(np.floor(Hout)), int(np.floor(Wout))


def layerShapeSeq(layerls,H0,W0):
    '''
    calculates output shape of sequence of layers
    (Hin,Win) -> ... -> (Hout,Wout)
    '''
    Hout,Wout = H0,W0
    for layer in layerls:
        if type(layer) == nn.Conv2d:
            Hout,Wout = convShape(layer,Hout,Wout)
        elif type(layer) == nn.MaxPool2d:
            Hout,Wout = poolShape(layer,Hout,Wout)
        else:
            raise NotImplementedError
    return Hout,Wout


def transformZconv(out):
    '''
    transforms 4d conv output to 2d matrix for ID prune,
    2nd index to be pruned
    '''
    out = out.reshape(-1, out.shape[1], np.prod(out.shape[2:]))
    out = out.permute(1,0,2)
    out = out.reshape(-1, np.prod(out.shape[1:]))
    out = out.T
    return out


def seqPartial(seqnn, target_idx, X):
    '''
    returns intermediate output of sequential list
    '''
    for i, layer in enumerate(seqnn):
        if i > target_idx: break
        X = layer(X)
    return X


def neuronnormNet(model, xtrain, norm="2"):
    with torch.no_grad():
        # normalize rows of fc1.weight
        # col because W is (h,d)
        if norm == "none":
            print("No neuron renorm")
            return
        elif norm == "1":
            colnorm = torch.sum(torch.abs(model.fc1.weight), axis=1)
        elif norm == "2":
            colnorm = torch.sqrt(torch.sum(torch.square(model.fc1.weight), axis=1))
        elif norm == "var":
            Z = torch.matmul(model.fc1.weight, xtrain.T)
            colnorm = 1 / torch.sqrt(torch.var(Z, axis=1))
        elif norm == "last1":
            #colnorm = 1 / torch.abs(model.fc2.weight).flatten()
            colnorm = 1 / torch.sum(torch.abs(model.fc2.weight), dim=0)
        elif norm == "last2":
            #colnorm = 1 / torch.square(model.fc2.weight).flatten()
            colnorm = 1 / torch.sqrt(torch.sum(torch.square(model.fc2.weight), dim=0))
        else:
            raise NotImplementedError

        model.fc1.weight = nn.Parameter(model.fc1.weight / colnorm[:,None])
        model.fc2.weight = nn.Parameter(model.fc2.weight * colnorm)
    return colnorm


def renormNet(model, X, colnorm, k):
    with torch.no_grad():
        Z = model.getZ(X).detach().numpy()
        k_idx, T = getID(k, Z, mode='kT')
        model.fc1.weight = nn.Parameter(model.fc1.weight * colnorm[k_idx][:,None])
        model.fc2.weight = nn.Parameter(model.fc2.weight / colnorm[k_idx])


def train(args, model, device, train_loader, criterion, optimizer, epoch, rand=False):
    model.train()
    avg_loss = 0
    correct_1, correct_5, total = 0, 0, 0
    for data, target in train_loader:
        #print(target.shape)

        data   = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
        if args.dataset == "fashion-mnist" or args.dataset == "cifar10":
            pred = output.argmax(dim=1, keepdim=True)  
            correct_1 += pred.eq(target.view_as(pred)).sum().item()
        elif args.dataset == "imagenet":
            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1,-1).expand_as(pred))
            correct_1 = correct[:1].view(-1).float().sum(0, keepdim=True)
            correct_5 = correct[:5].view(-1).float().sum(0, keepdim=True)
        else:
            raise NotImplementedError
        total += args.batch_size

    avg_loss /= len(train_loader) # over num batches
    if epoch % args.log_interval == 0:
        print('Train Epoch: {} \tAvg Loss: {:.6f} \tAccuracy: {}/{} ({:.1f}%) \tTop5 Accuracy: {}/{} ({:.1f}%'\
                .format(epoch, avg_loss, 
                correct_1, total,
                100. * correct_1 / total,
                correct_5, total,
                100. * correct_5 / total,
                flush=True))
    return avg_loss, correct_1 / total, correct_5 / total

def test(args, model, device, test_loader, criterion, epoch, returnAcc=False, pstatement="Test"):
    model.eval()
    test_loss = 0
    correct_1, correct_5, total = 0, 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data  = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            #data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            if args.dataset == "patches":
                test_loss += criterion(output, target).item()  
            elif args.dataset == "fashion-mnist" \
                    or args.dataset == "cifar10":
                test_loss += criterion(output, target).item()  
                # get the index of the max class
                pred = output.argmax(dim=1, keepdim=True)  
                correct_1 += pred.eq(target.view_as(pred)).sum().item()
            elif args.dataset == "imagenet":
                _, pred = output.topk(5, 1, True, True)
                pred = pred.t().to(device)
                correct = pred.eq(target.view(1,-1).expand_as(pred))
                correct_1 = correct[:1].view(-1).float().sum(0, keepdim=True)
                correct_5 = correct[:5].view(-1).float().sum(0, keepdim=True)
            else:
                raise NotImplementedError
            total += len(target)

    test_loss /= len(test_loader) # over num batches

    if epoch % args.log_interval == 0 and args.verbose:
        if args.dataset == "patches":
            print('{:s}: \t\t\tAvg Loss: {:f}\n'.format(pstatement, test_loss), flush=True)
        elif args.dataset == "fashion-mnist" or args. dataset == "cifar10":
            print('{:s}: Avg loss: {:f}, Accuracy: {}/{} ({:.1f}%) Top5 Accuracy: {}/{} ({:.1f}%)'.\
                    format(pstatement, 
                    test_loss, correct_1, total,
                    100. * correct_1 / total, 
                    correct_5, total,
                    100. * correct_5 / total),
                    flush=True)

    if returnAcc:
        return test_loss, correct_1 / total, correct_5 / total
    return test_loss


def load_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.arch == "FC1":
        state        = torch.load(args.load_fname)
        state        = state['model_state']
        in_feats     = state['fc1.weight'].shape[1]
        hidden_feats = state['fc1.weight'].shape[0]
        out_feats    = state['fc2.weight'].shape[0]
        model        = FC1(in_feats, hidden_feats, out_feats, False, True)
        model.load_state_dict(state)
    elif args.arch == "AlexNet":
        model = AlexNet()
        state = torch.load(args.load_fname, map_location=device)
        model.load_state_dict(state['model_state'])
    elif args.arch == "ResNet56":
        model = resnet_cifar(depth=56)
        state = torch.load(args.load_fname, map_location=device)
        model.load_state_dict(state['model_state'])
    elif args.arch == "VGG16":
        if args.dataset == "cifar10":
            model = vgg_cifar(depth=16)
            state = torch.load(args.load_fname, map_location=device)
            model.load_state_dict(state['model_state'])
        elif args.dataset == "imagenet":
            model = vgg16_imagenet(pretrained=True)
    elif args.arch == "ResNet50":
        model = resnet50_imagenet(pretrained=True)

    return model


def save_checkpoint(state, is_best, savename):
    torch.save(state, savename+'_checkpoint.pt')
    if is_best:
        shutil.copyfile(savename+'_checkpoint.pt', savename+'_best.pt')


def compare_pred(modelA, modelB, device, data_loader):
    AYBY, AYBN, ANBY, ANBN = 0, 0, 0, 0
    total = 0
    with torch.no_gard():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            total += len(target)
            outputA  = modelA(data)
            outputB  = modelB(data)
            predA    = outputA.argmax(dim=1, keepdim=True)
            predB    = outputB.argmax(dim=1, keepdim=True)
            correctA = predA.eq(target.view_as(predA)).sum().item()
            correctB = predB.eq(target.view_as(predB)).sum().item()

            AYBY += torch.sum(correctA == 1 and correctB == 1 
                    and correctA == correctB)
            AYBN += torch.sum(correctA == 1 and correctB == 0 
                    and correctA == correctB)
            ANBY += torch.sum(correctA == 0 and correctB == 1 
                    and correctA == correctB)
            ANBN += torch.sum(correctA == 0 and correctB == 0 
                    and correctA == correctB)

    return AYBY/total, AYBN/total, ANBY/total, ANBN/total


# Code from https://github.com/simochen/model-tools.

def print_model_param_nums(model, multiply_adds=True):
    total = sum([param.nelement() for param in model.parameters()])
    print(total)
    print('  + Number of params: %.2fM' % (total / 1e6))
    return total

def print_model_param_flops(model=None, input_res=224, multiply_adds=True):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = copy.deepcopy(model)
    foo(m)
    input = Variable(torch.rand(3, 3, input_res, input_res), requires_grad = True)
    input = input.to(device)
    out = m(input)

    print(list_conv, list_linear, list_bn, list_pooling, list_relu)
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))

    print('  + Number of FLOPs: %.5fG' % (total_flops / 3 / 1e9))

    return total_flops / 3, list_conv, list_linear
