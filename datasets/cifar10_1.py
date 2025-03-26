import os

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from detectron2.data import transforms as T

import random
from augmentations.augmentation import *
import pathlib
import json

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CIFAR10_1_datasets(Dataset):
    def __init__(self, is_train=True, target_img_w=32, target_img_h=32):

        data_path = "/home/hojunson/Desktop/Datasets/CIFAR/CIFAR-10.1-master/datasets"
        version_string = 'v6'
        filename = 'cifar10.1_' + version_string

        label_filename = filename + '_labels.npy'
        imagedata_filename = filename + '_data.npy'
        label_filepath = os.path.abspath(os.path.join(data_path, label_filename))
        imagedata_filepath = os.path.abspath(os.path.join(data_path, imagedata_filename))
        print('Loading labels from file {}'.format(label_filepath))
        assert pathlib.Path(label_filepath).is_file()
        self.labels = np.load(label_filepath)
        print('Loading image data from file {}'.format(imagedata_filepath))
        assert pathlib.Path(imagedata_filepath).is_file()
        self.imagedata = np.load(imagedata_filepath)
        assert len(self.labels.shape) == 1
        assert len(self.imagedata.shape) == 4
        assert self.labels.shape[0] == self.imagedata.shape[0]
        assert self.imagedata.shape[1] == 32
        assert self.imagedata.shape[2] == 32
        assert self.imagedata.shape[3] == 3

        self.is_train = is_train

        self.target_img_w = target_img_w
        self.target_img_h = target_img_h

        if self.is_train == True:
            self.aug_hflip = True
            self.aug_vflip = False
            self.aug_center_crop = False
            self.aug_crop = True
            self.aug_color_jitter = False
            self.aug_MIC = False
        else:
            self.aug_hflip = False
            self.aug_vflip = False
            self.aug_center_crop = False
            self.aug_crop = False
            self.aug_color_jitter = False
            self.aug_MIC = False

        self.augmentations = self.generate_augmentations()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.imagedata[idx]

        aug_input = T.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image, mask_img = aug_input.image, aug_input.sem_seg

        image = (image/255. - np.array((0.4914, 0.4822, 0.4465), dtype=np.float32))/ np.array((0.2023, 0.1994, 0.2010), dtype=np.float32) # normalize
        image = image.astype(np.float32)
        image = torch.as_tensor(image.copy().transpose(2, 0, 1))

        temp = dict()
        temp["image"] = image
        temp["label"] = self.labels[idx].astype(np.int64)

        return temp

    def generate_augmentations(self):

        augs = []
        augs.append(ShortestEdgeResize(self.target_img_h, interp=2))

        if self.is_train == True:

            if self.aug_center_crop == True:
                augs.append(CenterCropTransform([self.target_img_h, self.target_img_w]))

            if self.aug_crop == True:
                augs.append(RandomCropTransform((self.target_img_h, self.target_img_w),8, pad_if_needed=True))

            if self.aug_hflip == True:
                augs.append(T.RandomApply(T.RandomFlip(
                    horizontal=True,
                    vertical=False,
                ), prob=0.5))

            if self.aug_vflip == True:
                augs.append(T.RandomApply(T.RandomFlip(
                    horizontal=False,
                    vertical=True,
                ), prob=0.5))

            if self.aug_color_jitter == True:
                augs += [
                    T.RandomApply(T.AugmentationList([
                        T.RandomContrast(0.6, 1.4),
                        T.RandomBrightness(0.6, 1.4),
                        T.RandomSaturation(0.6, 1.4),
                    ]), prob=0.8),
                    T.RandomApply(T.RandomSaturation(0, 0), prob=0.2),  # Random grayscale
                    T.RandomApply(RandomBlurTransform((0.1, 2.0)), prob=0.5),
                ]

            if self.aug_MIC == True:
                augs.append(T.RandomApply(MICTransform(0.25, 48), prob=1.0))

        else:

            if self.aug_center_crop:
                augs.append(CenterCropTransform((self.target_img_h, self.target_img_w)))

        augmentations = T.AugmentationList(augs)
        return augmentations

if __name__ == "__main__":
    batch_size = 4
    load_mask = False
    training_data = CIFAR10_1_datasets(is_train=True,target_img_w=64, target_img_h=64)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=1)

    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # print labels
    import json

    for i, data in enumerate(train_dataloader, 0):
        print("i: ",i)

        temp = data
        images = temp["image"]
        labels = temp["label"]

        images = images * np.expand_dims(np.expand_dims(np.expand_dims(torch.as_tensor(np.array([0.5,0.5,0.5])), axis=0), axis=-1), axis=-1)
        images = images + np.expand_dims(np.expand_dims(np.expand_dims(torch.as_tensor(np.array([0.5,0.5,0.5])), axis=0), axis=-1), axis=-1)

        print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
        imshow(torchvision.utils.make_grid(images))