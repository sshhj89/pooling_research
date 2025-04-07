import os

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from detectron2.data import transforms as T

import random
from augmentations.augmentation import *
import meta_classes_dict

import json

class Imagenet_datasets(Dataset):
    def __init__(self, is_train=True, is_easy = True, load_mask=False, target_img_w=224, target_img_h=224):
        self.root_path = "/datasets/imagenet21k_resized"
        self.root_json_path = "/home/hojunson/PycharmProjects/pooling_research/"

        self.is_train = is_train
        if self.is_train == True:
            is_easy = False

        self.load_mask = load_mask

        if is_train == False:
            self.is_train = False
            self.load_mask = False

        if self.is_train == True:
            with open(os.path.join(self.root_json_path, "jsons/imagenet_sood_train_entire_segmentation.json"),
                      'r') as file:
                self.data = json.load(file)
        else:
            if is_easy == True:
                with open(os.path.join(self.root_json_path,
                                       "jsons/imagenet_sood_test_easy_ood_entire_classification.json"), 'r') as file:
                    self.data = json.load(file)
            else:
                with open(os.path.join(self.root_json_path,
                                       "jsons/imagenet_sood_test_hard_ood_entire_classification.json"), 'r') as file:
                    self.data = json.load(file)

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
            self.aug_center_crop = True
            self.aug_crop = False
            self.aug_color_jitter = False
            self.aug_MIC = False


        if self.is_train == True:
            self.num_classes = len(meta_classes_dict.sood_segmentation_id_to_name.keys())
            self.idx_classname = {str(v): meta_classes_dict.sood_segmentation_id_to_name[k] for k, v in
                                  meta_classes_dict.sood_segmentation_id_to_label.items()}
        else:
            self.num_classes = len(meta_classes_dict.sood_classification_id_to_name.keys())
            self.idx_classname = {str(v): meta_classes_dict.sood_classification_id_to_name[k] for k, v in
                                  meta_classes_dict.sood_classification_id_to_label.items()}

        self.augmentations = self.generate_augmentations()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx]
        img_path = data["image_path"]
        label = data["label"]
        mask_path = data["mask_path"]

        image = cv2.cvtColor(cv2.imread(os.path.join(self.root_path,img_path)), cv2.COLOR_BGR2RGB)
        mask_img = None

        if self.load_mask == True:
            if mask_path is not None and self.is_train == True:

                # print(os.path.join(self.root_path, mask_path))
                temp_mask = cv2.imread(os.path.join(self.root_path, mask_path), cv2.IMREAD_GRAYSCALE)
                temp_mask[temp_mask>0] = 1
                mask_img = np.expand_dims(temp_mask, axis=-1) #* 2

                aug_input = T.AugInput(image, sem_seg=mask_img)
                transforms = self.augmentations(aug_input)
                image, mask_img = aug_input.image, aug_input.sem_seg

            else:
                aug_input = T.AugInput(image, sem_seg=None)
                transforms = self.augmentations(aug_input)
                image, mask_img = aug_input.image, aug_input.sem_seg
        else:
            aug_input = T.AugInput(image, sem_seg=None)
            transforms = self.augmentations(aug_input)
            image, mask_img = aug_input.image, aug_input.sem_seg

        image = (image/255. - np.array([0.485, 0.456, 0.406], dtype=np.float32))/ np.array([0.229, 0.224, 0.225], dtype=np.float32) # normalize
        image = image.astype(np.float32)
        image = torch.as_tensor(image.copy().transpose(2, 0, 1))

        temp = dict()
        temp["image"] = image
        temp["image_path"] = img_path

        if self.is_train == True:
            temp["label"] = meta_classes_dict.sood_segmentation_id_to_label[label]
        else:
            temp["label"] = meta_classes_dict.sood_classification_id_to_label[label]

        if self.load_mask==True:
            if mask_img is not None:
                temp["mask_img"] = torch.as_tensor(mask_img.copy().transpose(2, 0, 1))
            else:
                temp["mask_img"] = torch.as_tensor(np.ones((1, self.target_img_h, self.target_img_w), dtype=np.uint8))

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
    load_mask = False
    training_data = Imagenet_datasets(is_train=False, is_easy=False, load_mask=load_mask, target_img_w=224, target_img_h=224)
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

        images = images * np.expand_dims(np.expand_dims(np.expand_dims(torch.as_tensor(np.array([0.229, 0.224, 0.225])), axis=0), axis=-1), axis=-1)
        images = images + np.expand_dims(np.expand_dims(np.expand_dims(torch.as_tensor(np.array([0.485, 0.456, 0.406])), axis=0), axis=-1), axis=-1)

        if load_mask == True:
            mask_images = temp["mask_img"]

        if load_mask == True:
            mask_images = temp["mask_img"]
            # imshow(torchvision.utils.make_grid(images*mask_images))
        else:
            # imshow(torchvision.utils.make_grid(images))
            pass