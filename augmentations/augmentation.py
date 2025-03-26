import random

import cv2
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from fvcore.transforms.transform import Transform, NoOpTransform
from scipy.ndimage import gaussian_filter


class RandomResizeCrop(Transform):
    def __init__(self, min_scale,max_scale,):
        super().__init__()
        self.size = output_size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

        self.crop_stx = 0
        self.crop_sty = 0

    def apply_image(self, img: np.ndarray) -> np.ndarray:

        pil_image = Image.fromarray(img)
        if self.padding > 0:
            img = F.pad(pil_image, self.padding)

            # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
            # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        img = np.ascontiguousarray(img)

        self.crop_sty = i
        self.crop_stx = j

        return img[i:i + self.size[0], j:j + self.size[1], :]

    def get_params(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """

        (h, w) = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:

        pil_image = Image.fromarray(np.squeeze(segmentation, axis=-1))
        if self.padding > 0:
            img = F.pad(pil_image, self.padding)

            # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
            # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        img = np.ascontiguousarray(img)

        return np.expand_dims(
            img[self.crop_sty:self.crop_sty + self.size[0], self.crop_stx:self.crop_stx + self.size[1]], axis=-1)

    def inverse(self) -> Transform:
        return NoOpTransform()

class RandomCropTransform(Transform):
    def __init__(self, output_size, padding=4, pad_if_needed=False):
        super().__init__()
        self.size = output_size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

        self.crop_stx = 0
        self.crop_sty = 0

    def apply_image(self, img: np.ndarray) -> np.ndarray:

        pil_image = Image.fromarray(img)

        if self.padding > 0:
            img = F.pad(pil_image, self.padding)

            # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
            # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        img = np.ascontiguousarray(img)

        self.crop_sty = i
        self.crop_stx = j

        temp = img[i:i + self.size[0], j:j + self.size[1], :]

        # print("temp.shape: ", temp.shape, i,j,i+h,j+w)

        return img[i:i + self.size[0], j:j + self.size[1], :]

    def get_params(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """

        (w,h) = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:

        pil_image = Image.fromarray(np.squeeze(segmentation, axis=-1))
        if self.padding > 0:
            img = F.pad(pil_image, self.padding)

            # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
            # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        img = np.ascontiguousarray(img)

        return np.expand_dims(
            img[self.crop_sty:self.crop_sty + self.size[0], self.crop_stx:self.crop_stx + self.size[1]], axis=-1)

    def inverse(self) -> Transform:
        return NoOpTransform()


class MICTransform(Transform):
    def __init__(self, ratio, block_size):
        super().__init__()
        self._set_attributes(locals())
        self.call_flag = 0
        self.input_mask = 0

    def apply_image(self, img: np.ndarray) -> np.ndarray:

        if len(img.shape) == 3:
            H, W, C = img.shape
        elif len(img.shape) == 2:  # binary masks
            H, W = img.shape

        was_int = False
        if img.dtype == np.uint8:
            was_int = True
            img = img.astype(np.float32)

        mh, mw = round(H / self.block_size), round(W / self.block_size)

        input_mask = np.random.rand(mh, mw)
        input_mask = input_mask > self.ratio
        self.input_mask = cv2.resize(np.asarray(input_mask, dtype="uint8"), (W, H), interpolation=cv2.INTER_NEAREST)

        masked_img = img * np.repeat(self.input_mask[..., np.newaxis], C, axis=-1)
        if was_int:
            return np.clip(masked_img, 0, 255).astype(np.uint8)
        else:
            return masked_img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        # return segmentation
        segmentation = segmentation * np.repeat(self.input_mask[..., np.newaxis], 1, axis=-1)

        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()


class RandomBlurTransform(Transform):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = gaussian_filter(img, sigma=sigma)
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            return gaussian_filter(img, sigma=sigma)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()


class CenterCropTransform(Transform):
    def __init__(self, output_size):
        super().__init__()

        self.crop_h = output_size[0]
        self.crop_w = output_size[1]

        self.size = output_size

        self.stx = 0
        self.sty = 0

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(img)

        return np.array(F.center_crop(pil_image, self.size))


    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:

        seg_image = np.squeeze(segmentation, axis=-1)

        pil_image = Image.fromarray(seg_image)

        return np.expand_dims(np.array(F.center_crop(pil_image, self.size)),axis=-1)


    def inverse(self) -> Transform:
        return NoOpTransform()

class ShortestEdgeResize(Transform):
    def __init__(self, output_size, max_size=None, antialias=True, interp=2):
        super().__init__()

        self.size = output_size

        self.interpolation = F._interpolation_modes_from_int(interp)
        self.interpolation_seg = F._interpolation_modes_from_int(1)
        self.max_size = max_size
        self.antialias = antialias

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(img)
        temp_img = F.resize(pil_image, self.size, self.interpolation, self.max_size, self.antialias)

        return np.array(temp_img)


    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:

        seg_image = np.squeeze(segmentation, axis=-1)
        pil_image = Image.fromarray(seg_image)
        temp_img = F.resize(pil_image, self.size, self.interpolation_seg, self.max_size, self.antialias)

        return np.expand_dims(np.array(temp_img),axis=-1)


    def inverse(self) -> Transform:
        return NoOpTransform()

