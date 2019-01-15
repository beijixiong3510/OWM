import argparse
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import scipy.io as scio
from myresnet import *
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # use gpu0,1


def create_dict(file):
    create_dict_ = {}
    handle = open(file)
    for line in handle:
        contents = line.split()
        create_dict_[contents[0]] = contents[1:]
    return create_dict_


def default_loader(img):
    return Image.open(img)


class custom_get_set(torch.utils.data.Dataset):
    def __init__(self, img_path, txt_path, img_transform=None, loader=default_loader):
        self.img_list = img_path
        self.label_list = txt_path
        self.img_transform = img_transform
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        label = np.array(label)
        img = self.loader(img_path)
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, label

    def __len__(self):
        return len(self.label_list)


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='./model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_false',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')


best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    model = myResNet.resnet50()
    print(model)
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.SmoothL1Loss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

#    optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    print("=> Data loading code...")
    dirPath = "/data0/celebA_PFC/img_celeba"
    dictAtt = create_dict('/data0/celebA_PFC/list_attr_celeba.txt')
    for key, values in dictAtt.items():
        try:
            pos = key.find('.jpeg')
            # valueList = values.split(',')
            for element in range(len(values)):
                if values[element] == '-1':
                    values[element] = 0
                else:
                    values[element] = 1
            dictAtt[key] = values  # ','.join(valueList)
        except:
            continue

    dictEval = create_dict('/data0/celebA_PFC/list_eval_partition.txt')
    imList = os.listdir(os.path.join(dirPath))
    trainFiles = []
    trainLabels = []
    # evalFiles = []
    # evalLabels = []
    testFiles = []
    testLabels = []
    for i_path in imList:
        partition = dictEval[i_path]
        if partition[0] == '0':
            trainFiles.append(os.path.join(dirPath, i_path))
            trainLabels.append(dictAtt[i_path])
        elif partition[0] == '1':  # eval
            trainFiles.append(os.path.join(dirPath, i_path))
            trainLabels.append(dictAtt[i_path])
        elif partition[0] == '2':
            testFiles.append(os.path.join(dirPath, i_path))
            testLabels.append(dictAtt[i_path])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = custom_get_set(trainFiles, trainLabels, train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = custom_get_set(testFiles, testLabels, test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print("=> Data loading Done!!!")

    if args.evaluate:
        validate(test_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(test_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).type(torch.cuda.FloatTensor)

        # compute output
        output, _ = model(input_var)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, _ = accuracy(output.data, target_var.data)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   top1=top1))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    acc_all = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    prec_all = 0
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True).type(torch.cuda.FloatTensor)

        # compute output
        output, x2 = model(input_var)
        save_data = False
        PATH = './data/celebA_mat50/train/'
        mat_name = 'celebAdata'+str(i)+'.mat'
        if save_data:
            os.makedirs(PATH, exist_ok=True)
            numpy_data = x2.data.cpu().numpy()
            numpy_lables = target_var.data.cpu().numpy()
            scio.savemat(os.path.join(PATH,  mat_name),
                         mdict={'data': numpy_data, 'lables': numpy_lables})

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec_each = accuracy(output.data, target_var.data)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))
        acc_all.update(prec_each, input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))
    #
    # numpy_data = acc_all.avg.cpu().numpy()
    # scio.savemat(os.path.join('./data/celebA_mat50/', 'accu.mat'),
    #              mdict={'accu': numpy_data})
    # print(acc_all.avg)
    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    correct = target.eq(torch.round(output))
    correct = correct.float()*100.0
    correct = torch.mean(correct, 0)
    res = torch.mean(correct)

    return res,correct


if __name__ == '__main__':
    main()
