from torchvision.models import vgg13, VGG13_Weights
import torch.nn as nn

import torch

# training_data = Imagenet_datasets(is_train=True, target_img_w=224, target_img_h=224)
#
# test_data_imagenet = Imagenet_datasets(is_train=False, target_img_w=224, target_img_h=224)
# test_data_loader = DataLoader(test_data_imagenet, batch_size=128,
#                                  shuffle=False, num_workers=16)

model = vgg13(weights=None)
print(model)

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG13(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG13, self).__init__()

        # Convolutional layers
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)

        # Conv block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)

        # Conv block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.maxpool(x)

        # Conv block 4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.maxpool(x)

        # Conv block 5
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = self.maxpool(x)

        # Flatten
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# Example usage:
model = VGG13(num_classes=1000)
print(model)

# model.eval()
#
# num_classes = 1000
# correct_pred = {str(idx): 0 for idx in range(0, num_classes)}
# total_pred = {str(idx): 0 for idx in range(0, num_classes)}
# device = torch.device('cuda:0')
# model.to(device)
#
# for data in test_data_loader:
#     images = data["images"]
#     labels = data["labels"]
#
#     images = images.to(device)
#     labels = labels.to(device)
#
#     outputs = model(images)
#
#     ''' accuracy per classes '''
#     _, predictions = torch.max(outputs, 1)
#     # collect the correct predictions for each class
#     for label, prediction in zip(labels, predictions):
#         # print(label)
#         if label == prediction:
#             temp_label = label.item()
#             correct_pred[str(temp_label)] += 1
#
#         temp_label = label.item()
#         total_pred[str(temp_label)] += 1
#
# # print accuracy for each class
# mean_accuracy = 0
# for pred_class, correct_count in correct_pred.items():
#     cls_accuracy = 100 * float(correct_count) / total_pred[pred_class]
#     mean_accuracy += cls_accuracy
#
#     classname = training_data.classes[int(pred_class)]
#     print(f'Accuracy for {classname:5s} is {cls_accuracy:.1f} %')
# print("top1 accuracy: ", (mean_accuracy / num_classes))