# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms

class BoundaryfunctionPooling2d(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        super(BoundaryfunctionPooling2d, self).__init__()


        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)
        # self.avg_pool = nn.AvgPool2d(kernel_size, stride, padding)

        # self.vis_trans = torchvision.transforms.ToPILImage()

    def forward(self, input, mask=None):

        if mask is not None:
            # print("barrier: ", torch.unique(mask))
            h, w = input.shape[2], input.shape[3]
            fg_mask = F.interpolate(mask, (h,w), mode="nearest")

            fg_count = F.avg_pool2d(fg_mask.float(), kernel_size=self.kernel_size,
                                    stride=self.stride, padding=self.padding) * (self.kernel_size ** 2)
            bg_count = (self.kernel_size ** 2) - fg_count
            pooled_max = self.max_pool(input)
            fg_dominant_mask = (fg_count > bg_count).float()
            output = fg_dominant_mask * pooled_max + (1 - fg_dominant_mask) * pooled_max

        else:
            output = self.max_pool(input)

        return output
