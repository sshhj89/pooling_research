import argparse
import os
from torch.utils.data import DataLoader
import time
from datetime import timedelta
import torch
import torchvision
import torchvision.transforms as transforms
import logging
import sys
import importlib
from importlib import import_module
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from collections import OrderedDict

import models.resnet.resnet_skip_max_gauss
from utils import save_checkpoint
from datasets.cifar10_1 import CIFAR10_1_datasets
from torchvision.models import resnet50, ResNet50_Weights

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default= 128, type=int,
                    metavar='N', help='mini-batch size (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--device', '-d', default=0, type=int,
                    metavar='N', help='gpu number')
parser.add_argument('--ptype', dest='ptype', default="skip", # "skip": original resnet18. "gauss_CN", "max"
                    help='pooling type')
parser.add_argument('--dataset', dest='dataset', type=str, default="CIFAR10", #Imagenet(Imagenet-S), Coco, Pascal, Imagenet_car
                    help='dataset type')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)

''' to do
    cifar 100
    different model training 
    different pooling 
    apply LIft and multi-proxy pooling
'''

def main():
    ''' ptype: max and use_mask: True means training with foreground focus
        ptype: skip and use_mask: anything means original resnet18
        adjust last pool to control adaptive max and adaptive avg pooling at the last layer.

        python main_train.py --dataset Imagenet --use-avg-pool --mask-ratio 0.6 --ptype skip --print-freq 10 --device 0 --epochs 250 --batch-size 256 --lr 0.05 |& tee -a log.txt

        to use max pool for the last adaptive pool, remove --use-avg-pool.
        if you want to use mask during training, add --use-mask.
    '''

    num_input_sz = 32
    global args, best_prec
    args = parser.parse_args()

    args.save_dir = ("saved_{}_dataset_{}_ptype_{}".format(args.arch, args.dataset, args.ptype))

    print("args.save_dir: ", args.save_dir)
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        print("generate: ", args.save_dir)
        os.makedirs(args.save_dir)

    print("architecture: ", args.arch)
    print("ptype: ", args.ptype)
    print("args: ", args)

    logging_name = ("model_{}_dataset_{}_ptype_{}".format(args.arch, args.dataset, args.ptype))

    logger = logging.getLogger(logging_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    log_filename = os.path.join(args.save_dir, "log.txt")

    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    screen_handler = logging.StreamHandler(stream=sys.stdout)  # stream=sys.stdout is similar to normal print
    logger.addHandler(screen_handler)

    logger.info(logging_name)
    logger.info("args: "+args.__str__())

    device = torch.device('cuda:' + str(args.device))

    training_data = None

    if args.dataset == "CIFAR10":
        path_cifar10 = "/home/hojunson/Desktop/Datasets/CIFAR/cifar-10-python/"

        transform_train = transforms.Compose([
                transforms.Resize(num_input_sz),
                transforms.RandomCrop(num_input_sz, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        training_data = torchvision.datasets.CIFAR10(root=path_cifar10, train=True,
                                                download=False, transform=transform_train)
    elif args.dataset == "CIFAR100":
        path_cifar100 = "/home/hojunson/Desktop/Datasets/CIFAR/cifar-100-python/"


        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        transform_train = transforms.Compose([
            transforms.Resize(num_input_sz),
            transforms.RandomCrop(num_input_sz, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])

        training_data = torchvision.datasets.CIFAR100(root=path_cifar100, train=True,
                                                     download=False, transform=transform_train)
    else:
        assert 1 != 1, print("no test data is set")

    test_data_cifar10 = None
    test_data_cifar10_1 = None
    test_data_cifar100 = None

    if args.dataset == "CIFAR10":
        path_cifar10 = "/home/hojunson/Desktop/Datasets/CIFAR/cifar-10-python/"

        transform = transforms.Compose([
            transforms.Resize(num_input_sz),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_data_cifar10 = torchvision.datasets.CIFAR10(root=path_cifar10, train=False,
                                               download=False, transform=transform)

        test_data_cifar10_1 = CIFAR10_1_datasets(is_train=False, target_img_w=num_input_sz, target_img_h=num_input_sz)

    elif args.dataset == "CIFAR100":
        path_cifar100 = "/home/hojunson/Desktop/Datasets/CIFAR/cifar-100-python/"

        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        transform_test = transforms.Compose([
            transforms.Resize(num_input_sz),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])

        test_data_cifar100 = torchvision.datasets.CIFAR100(root=path_cifar100, train=False, download=False, transform=transform_test)

    else:
        assert 1!=1, print("no test data is set")

    trainloader = None

    if training_data is not None:
        trainloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                 pin_memory=True)

    num_classes = len(training_data.classes)
    print("num_classes: ", num_classes)

    test_loaders = OrderedDict()
    if args.dataset == "CIFAR10":
        testloader_cifar10 = DataLoader(test_data_cifar10, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=args.workers)
        testloader_cifar10_1 = DataLoader(test_data_cifar10_1, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        test_loaders['cifar10'] = testloader_cifar10
        test_loaders['cifar10_1'] = testloader_cifar10_1

    elif args.dataset == "CIFAR100":
        testloader_cifar100 = DataLoader(test_data_cifar100, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.workers)

        test_loaders['cifar100'] = testloader_cifar100

    model = None
    # New weights with accuracy 80.858%
    model = resnet50(weights=None)
    model.fc = nn.Linear(2048, 10)

    assert model is not None, "need to check model architecture"
    model.to(device)

    logger.info(model.__str__())

    # print("model: ", model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*1.1)

    # grad_scaler = GradScaler()

    best_prec_cifar10 = 0.
    best_prec_cifar10_1 = 0.
    best_prec_cifar100 = 0.

    for epoch in range(0, args.epochs):  # loop over the dataset multiple times
        # adjust_learning_rate(optimizer, epoch, args)

        model.train()
        start = time.time()

        train_loss = 0

        last_batch_idx = 0
        for batch_idx, data in enumerate(trainloader):

            images = data[0]
            labels = data[1]
            images = images.to(device)
            labels = labels.to(device)

            # with torch.autocast('cuda', dtype=torch.float16):
            #     outputs = model(images)
            #     loss = criterion(outputs, labels)

            #     ''' auto cast'''
            #     optimizer.zero_grad()
            #     grad_scaler.scale(loss).backward()
            #     grad_scaler.step(optimizer)
            #     grad_scaler.update()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            last_batch_idx = batch_idx

        end = time.time()
        time_delta = str(timedelta(seconds=((end - start) * (args.epochs - epoch))))
        curr_lr = lr_scheduler.get_last_lr()[0]
        logger.info(f'Epoch: [{epoch}/{args.epochs}] estm time to finish {time_delta}, loss: {train_loss/(last_batch_idx+1):.5f}, lr: {curr_lr:.10f}')

        if epoch % args.print_freq == 0:
            model.eval()

            with torch.no_grad():

                for data_name, test_data_loader in test_loaders.items():

                    correct_pred = {str(idx): 0 for idx in range(0,num_classes)}
                    total_pred = {str(idx): 0 for idx in range(0,num_classes)}

                    for data in test_data_loader:
                        if 'cifar10_1' in data_name:
                            images = data['image']
                            labels = data['label']
                        else:
                            images = data[0]
                            labels = data[1]

                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = model(images)

                        ''' accuracy per classes '''
                        _, predictions = torch.max(outputs, 1)
                        # collect the correct predictions for each class
                        for label, prediction in zip(labels, predictions):
                            # print(label)
                            if label == prediction:
                                temp_label = label.item()
                                correct_pred[str(temp_label)] += 1

                            temp_label = label.item()
                            total_pred[str(temp_label)] += 1

                    # print accuracy for each class
                    mean_accuracy = 0
                    for pred_class, correct_count in correct_pred.items():
                        cls_accuracy = 100 * float(correct_count) / total_pred[pred_class]
                        mean_accuracy += cls_accuracy

                        classname = training_data.classes[int(pred_class)]
                        logger.info(f'Accuracy for {classname:5s} is {cls_accuracy:.1f} %')

                    if data_name == "cifar10":
                        if best_prec_cifar10 < (mean_accuracy / num_classes):
                            best_prec_cifar10 = max(mean_accuracy / num_classes, best_prec_cifar10)

                            save_checkpoint({
                                'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'best_prec': best_prec_cifar10,
                            }, filename=os.path.join(args.save_dir, 'cifar10_checkpoint_{}.tar'.format(epoch)))
                        logger.info(f'Accuracy of the network on the {data_name} images: {best_prec_cifar10:.2f} %')

                    elif data_name == "cifar10_1":
                        if best_prec_cifar10_1 < (mean_accuracy / num_classes):
                            best_prec_cifar10_1 = max(mean_accuracy / num_classes, best_prec_cifar10_1)

                            save_checkpoint({
                                'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'best_prec': best_prec_cifar10_1,
                            }, filename=os.path.join(args.save_dir, 'cifar10_1_checkpoint_{}.tar'.format(epoch)))

                        logger.info(f'Accuracy of the network on the {data_name} images: {best_prec_cifar10_1:.2f} %')

                    elif data_name == "cifar100":
                        if best_prec_cifar100 < (mean_accuracy / num_classes):
                            best_prec_cifar100 = max(mean_accuracy / num_classes, best_prec_cifar100)

                            save_checkpoint({
                                'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'best_prec': best_prec_cifar100,
                            }, filename=os.path.join(args.save_dir, 'cifar100_checkpoint_{}.tar'.format(epoch)))
                        logger.info(f'Accuracy of the network on the {data_name} images: {best_prec_cifar100:.2f} %')

        lr_scheduler.step()

    if args.dataset == "CIFAR10":
        logger.info(f"Final best_prec_cifar10: {best_prec_cifar10:.2f} %, best_prec_cifar10_1: {best_prec_cifar10_1:.2f} %")
    elif args.dataset == "CIFAR100":
        logger.info(f"Final best_prec_cifar100: {best_prec_cifar100:.2f} %")

def adjust_learning_rate(optimizer, epoch, args):

    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 40))
    print("epoch {}: curr lr: {}".format(epoch, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()