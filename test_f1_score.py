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
from torchmetrics.classification import MulticlassF1Score

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
    print("architecture: ", args.arch)
    print("ptype: ", args.ptype)

    test_data_easy = Imagenet_ood.Imagenet_datasets(is_train=False, is_easy=True, load_mask=False, target_img_w=224,target_img_h=224)
    test_data_hard = Imagenet_ood.Imagenet_datasets(is_train=False, is_easy=False, load_mask=False, target_img_w=224,target_img_h=224)

    testloader_easy = DataLoader(test_data_easy, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    testloader_hard = DataLoader(test_data_hard, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

    training_data = Imagenet_ood.Imagenet_datasets(is_train=True, is_easy=False, load_mask=True, target_img_w=224,
                                                   target_img_h=224)

    num_classes = training_data.num_classes
    print("num_classes: ",num_classes)

    model = None
    if "resnet18" in args.arch:
        if "max" in args.ptype or "skip" in args.ptype:
            try:
                model_path = 'models.resnet.resnet_skip_max_gauss'
                model = eval(model_path).resnet18(pretrained=False, progress=True, num_classes=num_classes,
                                                  ptype=args.ptype, use_adaptmax=False)
            except ImportError:
                print('the network name you have entered is not supported yet')
                sys.exit()
        elif "gauss" in args.ptype:
            try:
                model_path = 'models.resnet.resnet_skip_max_gauss'
                model = eval(model_path).resnet18(pretrained=False, progress=True, num_classes=num_classes,
                                                  ptype=args.ptype, use_adaptmax=False)
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

    device = torch.device('cuda:' + str(args.device))
    model.to(device)

    checkpoint = torch.load('saved_models/saved_resnet18_dataset_Imagenet_ood_ptype_boundary_avgpool_2025-04-07 03:19:00/easy_checkpoint_90.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    f1_score_easy = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)  # or 'micro', 'weighted'
    f1_score_hard = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)  # or 'micro', 'weighted'

    correct_pred = {str(idx): 0 for idx in range(0, num_classes)}
    total_pred = {str(idx): 0 for idx in range(0, num_classes)}

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

            '''f1 scores'''
            f1_score_easy.update(predictions, labels)

    # print accuracy for each class
    mean_accuracy = 0
    for classname, correct_count in correct_pred.items():
        cls_accuracy = 100 * float(correct_count) / total_pred[classname]
        mean_accuracy += cls_accuracy

        classname = test_data_easy.idx_classname[classname]
        logger.info(f'Accuracy for class: {classname:5s} is {cls_accuracy:.2f} %')

    print(f'Accuracy of the network on the test_easy images: {(mean_accuracy / num_classes):.2f} %')

    score = f1_score_easy.compute()
    print(f"F1 Score (macro avg) easy: {score.item():.4f}")

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

            '''f1 scores'''
            f1_score_hard.update(predictions, labels)

    # print accuracy for each class
    mean_accuracy = 0
    for classname, correct_count in correct_pred.items():
        cls_accuracy = 100 * float(correct_count) / total_pred[classname]
        mean_accuracy += cls_accuracy

        classname = test_data_hard.idx_classname[classname]
        logger.info(f'Accuracy for class: {classname:5s} is {cls_accuracy:.2f} %')

    print(f'Accuracy of the network on the test_hard images: {(mean_accuracy / num_classes):.2f} %')

    score = f1_score_hard.compute()
    print(f"F1 Score (macro avg) hard: {score.item():.4f}")

if __name__ == '__main__':
    main()