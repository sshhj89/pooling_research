import os

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from detectron2.data import transforms as T
from torchvision.transforms import transforms

import random
from augmentations.augmentation import *
import meta_classes_dict

import json

class Imagenet_datasets(Dataset):
    def __init__(self, is_train=True,  target_img_w=224, target_img_h=224,):
        self.root_path = "/datasets"
        self.root_mask_path = self.root_path
        self.root_json_path = "/home/hojunson/PycharmProjects/Pooling_research/"

        self.is_train = is_train

        if self.is_train == True:
            with open(os.path.join(self.root_json_path,"jsons/imagenet_train.json"), 'r') as file:
                self.data = json.load(file)

            self.root_path = os.path.join( self.root_path,"Imagenet","train")

        else:
            with open(os.path.join(self.root_json_path,"jsons/imagenet_val.json"), 'r') as file:
                self.data = json.load(file)
            self.root_path = os.path.join(self.root_path, "Imagenet", "val")

        self.target_img_w = target_img_w
        self.target_img_h = target_img_h

        if self.is_train == True:
            self.aug_hflip = True
            self.aug_vflip = False
            self.aug_center_crop = True
            self.aug_crop = False
            self.aug_color_jitter = True
            self.aug_MIC = False
        else:
            self.aug_hflip = False
            self.aug_vflip = False
            self.aug_center_crop = True
            self.aug_crop = False
            self.aug_color_jitter = False
            self.aug_MIC = False

        temp = json.load(open("/home/hojunson/PycharmProjects/pooling_research/jsons/imagenet_class_index.json"))
        self.classes = [a[1] for a in temp.values()]

        self.augmentations = self.generate_augmentations()

        self.train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(target_img_w),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(target_img_w),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx]
        img_path = data["image"]
        label = data["label"]

        # print(img_path)

        # image = Image.open(os.path.join(self.root_path, img_path))
        image = cv2.cvtColor(cv2.imread(os.path.join(self.root_path, img_path)), cv2.IMREAD_COLOR)
        if self.is_train == True:
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)

        # image = cv2.cvtColor(cv2.imread(os.path.join(self.root_path, img_path)), cv2.COLOR_BGR2RGB)
        # aug_input = T.AugInput(image, sem_seg=None)
        # transforms = self.augmentations(aug_input)
        # image, mask_img = aug_input.image, aug_input.sem_seg
        #
        # image = (image/255. - np.array([0.485, 0.456, 0.406], dtype=np.float32))/ np.array([0.229, 0.224, 0.225], dtype=np.float32) # normalize
        # image = image.astype(np.float32)
        # image = torch.as_tensor(image.copy().transpose(2, 0, 1))
        # print(image.shape)
        temp = dict()
        temp["images"] = image
        temp["images_path"] = img_path
        temp["labels"] = label

        return temp
    def generate_augmentations(self):

        augs = []
        augs.append(ShortestEdgeResize(self.target_img_h, interp=2))

        if self.is_train == True:

            if self.aug_center_crop == True:
                augs.append(CenterCropTransform([self.target_img_h, self.target_img_w]))

            if self.aug_crop == True:
                augs.append(RandomCropTransform((self.target_img_h, self.target_img_w),32, pad_if_needed=True))

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
    training_data = Imagenet_datasets(is_train=False,  target_img_w=224, target_img_h=224)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=1)

    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # print labels
    import json

    class_idx = json.load(open("../jsons/imagenet_class_index.json"))

    # classes = ('car', 'person')

    for i, data in enumerate(train_dataloader, 0):
        print("i: ",i)

        temp = data
        images = temp["images"]
        labels = temp["labels"]

        images = images * np.expand_dims(np.expand_dims(np.expand_dims(torch.as_tensor(np.array([0.229, 0.224, 0.225])), axis=0), axis=-1), axis=-1)
        images = images + np.expand_dims(np.expand_dims(np.expand_dims(torch.as_tensor(np.array([0.485, 0.456, 0.406])), axis=0), axis=-1), axis=-1)
        # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

        imshow(torchvision.utils.make_grid(images))