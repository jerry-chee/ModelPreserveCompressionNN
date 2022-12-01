import time
import copy
import numpy as np
import torch
import torchsummary
from torch.autograd import Variable
from enum import Enum
from vgg_imagenet import vgg16_imagenet
#from resnet_imagenet import resnet50_imagenet

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


def load_model(args, pretrain=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {args.arch} architecture")
    print(f"using {args.load_fname} load_fname")
    print(f"using pretrain: {pretrain}")
    if (args.arch == "VGG16") or (args.arch == "vgg16"):
        if pretrain:
            model = vgg16_imagenet(pretrained=True)
        else:
        #TODO 
        #if args.pruner == "nonepre":
            print("loading model from ", args.load_fname)
            model = vgg16_imagenet(pretrained=False, 
                    prune_reload=True, prune_k=args.k)
            state = torch.load(args.load_fname, map_location=device)
            model.load_state_dict(state['model_state'])
    elif args.arch == "ResNet50":
        raise NotImplementedError
        #if pretrain:
        #    model = resnet50_imagenet(pretrained=True)
        #else:
        #    raise NotImplementedError

    return model

def modify_torch_pickle(fname, save_separate=False):
    '''
    change torch save format to pickle format, bc torchprune giving error with torch format
    not do not run distributed!
    '''
    import shutil
    import os
    import pickle
    import torch
    import torch.distributed as dist
    import torchprune as tp

    # first make copy of fname
    shutil.copyfile(fname, fname+"_torchbkup")
    # setup ddp
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=0, world_size=1)

    checkpoint = torch.load(fname, map_location='cpu')
    if save_separate:
        shutil.move(fname, fname+"_torchbkup")
        fname_new = fname.replace(".pth.tar", "")
        # save stats
        stats = {
            'epoch': checkpoint['epoch'],
            'arch': checkpoint['arch'],
            'best_acc1': checkpoint['best_acc1'],
        }
        torch.save(stats, fname_new+"_stats.pth.tar")
        #with open(fname_new+"_stats.pth.tar", 'wb') as f:
        #    pickle.dump(stats, f)
        # save state_dict
        torch.save(checkpoint['state_dict'], fname_new+"_statedict.pth.tar")
        #with open(fname_new+"_statedict.pth.tar", 'wb') as f:
        #    pickle.dump(checkpoint['state_dict'], f)
        # save model
        torch.save(checkpoint['model'].module, fname_new+"_model.pth.tar")
        #with open(fname_new+"_model.pth.tar", 'wb') as f:
        #    pickle.dump(checkpoint['model'].module, f)
        # save optimizer
        torch.save(checkpoint['optimizer'], fname_new+"_optimizer.pth.tar")
        #with open(fname_new+"_optimizer.pth.tar", 'wb') as f:
        #    pickle.dump(checkpoint['optimizer'], f)
    else:
        checkpoint['model'] = checkpoint['model'].module
        torch.save(checkpoint, fname)
        #with open(fname, 'wb') as f:
        #    pickle.dump(checkpoint, f)


class StatsMeter(object):
    def __init__(self, idstr):
        self.meter = np.zeros(0)
        self.cnt = 0
        self.idstr = idstr

    def update(self, new):
        self.meter = np.append(self.meter, new)
        self.cnt += 1

    def calc_stats(self):
        # mean and std
        self.meter_mean  = np.mean(self.meter)
        self.meter_std   = np.std(self.meter)
        print("{:s}".format(self.idstr))
        print("meter mean:{:f}".format(self.meter_mean))
        print("meter std:{:f}".format(self.meter_std))

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return top1.avg

def test(model_id, model_full, device, test_loader, criterion, epoch, calc_corr=True):
    model_id.eval()
    if calc_corr: model_full.eval()
    test_loss = 0
    correct_1, correct_5, match, total = 0, 0, 0, 0
    timer = StatsMeter(idstr="Timer")
    with torch.no_grad():
        for data, target in test_loader:
            data  = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            inf_start = time.time()
            output = model_id(data)
            if calc_corr:
                target_full = model_full(data)
            test_loss += criterion(output, target).item()  
            inf_end = time.time()
            timer.update(inf_end-inf_start)
            # sum up batch loss
            _, pred = output.topk(5, 1, True, True)
            pred = pred.t().to(device)
            correct = pred.eq(target.view(1,-1).expand_as(pred))
            correct_1 += correct[:1].view(-1).float().sum(0, keepdim=True).item()
            correct_5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
            total += len(target)
            # match outputs to original, top-1
            if calc_corr:
                match += (output.argmax(dim=1) == target_full.argmax(dim=1)).sum().item()

    test_loss /= len(test_loader) # over num batches

    print(f"Epoch: {epoch}\n \
            Test loss: {test_loss}\n \
            Accuracy: {correct_1/total} ({correct_1} / {total})\n \
            Top5 Accuracy: {correct_5/total} ({correct_5} / {total})\n \
            Corr: {match/total} ({match} / {total})")
    #print("Average test loss: ", test_loss)
    #print("Accuracy: ", (correct_1 / total))
    #print("correct_1: ", correct_1)
    #print("total: ", total)
    timer.calc_stats()
    return test_loss, correct_1 / total, correct_5 / total

def flopsfoo(net):
    childrens = list(net.children())
    if not childrens:
        if isinstance(net, torch.nn.Conv2d):
            print("Conv2d")
        elif isinstance(net, torch.nn.Linear):
            print("Linear")
        elif isinstance(net, torch.nn.BatchNorm2d):
            print("BatchNorm2d")
        elif isinstance(net, torch.nn.ReLU):
            print("ReLU")
        elif isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
            print("Max")
        elif isinstance(net, torch.nn.Upsample):
            print("Upsample")
        else:
            raise NotImplementedError
            #print(net)
        return
    for c in childrens:
        flopsfoo(c)


def model_summary(modelA, modelB=None, strA='', strB='',
        dataset='cifar10', summary_input=None, input_res=None):
    #print(strA)
    #torchsummary.summary(modelA, summary_input)
    if modelB is not None:
        print(strB)
        torchsummary.summary(modelB, summary_input)
    if dataset == 'cifar10' or dataset == 'imagenet':
        print_model_param_nums(modelA)
        if modelB is not None:
            print_model_param_nums(modelB)
        return(print_model_param_flops(modelA, input_res))
        if modelB is not None:
            print(print_model_param_flops(modelB, input_res))
    

# Code from https://github.com/simochen/model-tools.

def print_model_param_nums(model, multiply_adds=True):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

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
        if (self.bias is not None) or (self.bias):
            bias_ops = self.bias.nelement()
        else:
            bias_ops = 0.0

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

    #print(list_conv, list_linear, list_bn, list_pooling, list_relu)
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))

    print('  + Number of FLOPs: %.5fG' % (total_flops / 3 / 1e9))

    return total_flops / 3, list_conv, list_linear
