import argparse
import os

import sys
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
import models.resnet.resnet_skip_max_gauss
import models.resnet.resnet_boundary

from datasets import Imagenet_ood
import time
from datetime import timedelta
from utils import save_checkpoint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import time
from warmup_scheduler_pytorch import WarmUpScheduler
import logging

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
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
parser.add_argument('--ptype', dest='ptype', default="boundary", # "skip": original resnet18. "gauss_CN", "max"
                    help='pooling type')
parser.add_argument('--dataset', dest='dataset', type=str, default="Imagenet_ood", #Imagenet_ood_sampled #Imagenet(Imagenet-S), Coco, Pascal, Imagenet_car
                    help='dataset type')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)

def main():
    ''' ptype: max and use_mask: True means training with foreground focus
        ptype: skip and use_mask: anything means original resnet18
        adjust last pool to control adaptive max and adaptive avg pooling at the last layer.

        python main_train_gauss_max.py --dataset Imagenet_ood  --ptype max --print-freq 10 --device 0 --epochs 250 --batch-size 256 --lr 0.05 |& tee -a log.txt

        to use max pool for the last adaptive pool, remove --use-avg-pool.
        if you want to use mask during training, add --use-mask.
    '''

    args = parser.parse_args()

    if args.save_dir == "save_temp":
        args.save_dir = ("saved_models/saved_{}_dataset_{}_ptype_{}".format(args.arch, args.dataset,
                                                               args.ptype)) + "_" + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                                  time.gmtime(
                                                                                                      time.time()))

    print("args.save_dir: ", args.save_dir)
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        print("generate: ", args.save_dir)
        os.makedirs(args.save_dir)

    print("architecture: ", args.arch)
    print("ptype: ", args.ptype)
    # print("args: ", args)

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
    logger.info("args: " + args.__str__())

    device = torch.device('cuda:' + str(args.device))

    training_data = Imagenet_ood.Imagenet_datasets(is_train=True, is_easy=False, load_mask=True, target_img_w=224, target_img_h=224)
    test_data_easy = Imagenet_ood.Imagenet_datasets(is_train=False, is_easy=True, load_mask=False, target_img_w=224,target_img_h=224)
    test_data_hard = Imagenet_ood.Imagenet_datasets(is_train=False, is_easy=False, load_mask=False, target_img_w=224,target_img_h=224)

    trainloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                             pin_memory=True)

    testloader_easy = DataLoader(test_data_easy, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    testloader_hard = DataLoader(test_data_hard, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

    num_classes = training_data.num_classes
    print("num_classes: ",num_classes)

    model = None
    if "resnet18" in args.arch:
        if "max" in args.ptype or "skip" in args.ptype:
            try:
                model_path = 'models.resnet.resnet_skip_max_gauss'
                model = eval(model_path).resnet18(pretrained=False, progress=True, num_classes=num_classes,
                                                  ptype=args.ptype)
            except ImportError:
                print('the network name you have entered is not supported yet')
                sys.exit()
        elif "gauss" in args.ptype:
            try:
                model_path = 'models.resnet.resnet_skip_max_gauss'
                model = eval(model_path).resnet18(pretrained=False, progress=True, num_classes=num_classes,
                                                  ptype=args.ptype)
            except ImportError:
                print('the network name you have entered is not supported yet')
                sys.exit()
        elif "boundary" in args.ptype:
            try:
                model_path = 'models.resnet.resnet_boundary'
                model = eval(model_path).resnet18(pretrained=False, progress=True, num_classes=num_classes,
                                                  ptype=args.ptype)
            except ImportError:
                print('the network name you have entered is not supported yet')
                sys.exit()

    assert model is not None, "need to check model architecture"
    model.to(device)

    logger.info(model.__str__())

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = MultiStepLR(optimizer, milestones=[50, 85, 110], gamma=0.1)
    warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler, len_loader=len(trainloader),
                                       warmup_steps=2500, warmup_mode="linear", warmup_start_lr=0.00025)

    grad_scaler = GradScaler()

    best_prec_easy = 0.0
    best_prec_hard = 0.0

    for epoch in range(0, args.epochs):  # loop over the dataset multiple times

        model.train()
        start = time.time()

        train_loss = 0
        last_batch_idx = 0

        for batch_idx, data in enumerate(trainloader):
            images = data["image"]
            labels = data["label"]
            mask_images = data["mask_img"]

            images = images.to(device)
            labels = labels.to(device)
            mask_images = mask_images.to(device)

            with torch.autocast('cuda', dtype=torch.float16):
                if args.ptype == "boundary":
                    outputs = model(images, mask_images)
                else:
                    outputs = model(images)

                loss = criterion(outputs, labels)

                ''' auto cast'''
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                train_loss += loss.item()

            warmup_scheduler.step()

            last_batch_idx = batch_idx
            # optimizer.zero_grad()
            #
            # outputs = model(images)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()

        end = time.time()
        time_delta = str(timedelta(seconds=((end - start) * (args.epochs - epoch))))
        curr_lr = lr_scheduler.get_last_lr()[0]

        logger.info(
            f'Epoch: [{epoch}/{args.epochs}] estm time to finish {time_delta}, loss: {train_loss / (last_batch_idx + 1):.5f}, lr: {curr_lr:.10f}')

        if epoch % args.print_freq == 0:
            model.eval()

            correct_pred = {str(idx): 0 for idx in range(0,num_classes)}
            total_pred = {str(idx): 0 for idx in range(0,num_classes)}

            with torch.no_grad():
                for data in testloader_easy:
                    images = data["image"]
                    labels = data["label"]
                    # mask_images = data["mask_img"]

                    images = images.to(device)
                    labels = labels.to(device)

                    if args.ptype == "boundary":
                        outputs = model(images, None)
                    else:
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
            for classname, correct_count in correct_pred.items():
                cls_accuracy = 100 * float(correct_count) / total_pred[classname]
                mean_accuracy += cls_accuracy

                classname = test_data_easy.idx_classname[classname]
                logger.info(f'Accuracy for class: {classname:5s} is {cls_accuracy:.2f} %')

            logger.info(f'Accuracy of the network on the test_easy images: {(mean_accuracy / num_classes):.2f} %')

            if best_prec_easy < (mean_accuracy / num_classes):
                best_prec_easy = max(mean_accuracy / num_classes, best_prec_easy)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec': best_prec_easy,
                }, filename=os.path.join(args.save_dir, 'easy_checkpoint_{}.tar'.format(epoch)))

            logger.info(f"curr best_prec_easy on test easy:  {best_prec_easy:.2f} %")

            correct_pred = {str(idx): 0 for idx in range(0, num_classes)}
            total_pred = {str(idx): 0 for idx in range(0, num_classes)}

            with torch.no_grad():
                for data in testloader_hard:
                    images = data["image"]
                    labels = data["label"]
                    # mask_images = data["mask_imgs"]

                    images = images.to(device)
                    labels = labels.to(device)

                    if args.ptype == "boundary":
                        outputs = model(images, None)
                    else:
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
            for classname, correct_count in correct_pred.items():
                cls_accuracy = 100 * float(correct_count) / total_pred[classname]
                mean_accuracy += cls_accuracy

                classname = test_data_hard.idx_classname[classname]
                logger.info(f'Accuracy for class: {classname:5s} is {cls_accuracy:.2f} %')

            logger.info(f'Accuracy of the network on the test_hard images: {(mean_accuracy / num_classes):.2f} %')

            if best_prec_hard < (mean_accuracy / num_classes):
                best_prec_hard = max(mean_accuracy / num_classes, best_prec_hard)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec': best_prec_hard,
                }, filename=os.path.join(args.save_dir, 'hard_checkpoint_{}.tar'.format(epoch)))

            logger.info(f"curr best_prec on test_hard: {best_prec_hard:.2f} ")

    logger.info(f"Final best_prec_easy best_prec_hard: , {best_prec_easy:.2f}%, {best_prec_hard:.2f}%")

if __name__ == '__main__':
    main()