from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
import torch.nn as nn
from datasets.Imagenet import Imagenet_datasets
from torch.utils.data import DataLoader
import torch

model2 = resnet18(weights=None)
print(model2)

# New weights with accuracy 80.858%
# model2 = resnet50(weights=None)
#
# model2.fc = nn.Linear(2048, 10)
#
#
# print(model2)

training_data = Imagenet_datasets(is_train=True, target_img_w=224, target_img_h=224)

test_data_imagenet = Imagenet_datasets(is_train=False, target_img_w=224, target_img_h=224)
test_data_loader = DataLoader(test_data_imagenet, batch_size=128,
                                 shuffle=False, num_workers=16)

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

model.eval()

num_classes = 1000
correct_pred = {str(idx): 0 for idx in range(0, num_classes)}
total_pred = {str(idx): 0 for idx in range(0, num_classes)}
device = torch.device('cuda:0')
model.to(device)

for data in test_data_loader:
    images = data["images"]
    labels = data["labels"]

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
    print(f'Accuracy for {classname:5s} is {cls_accuracy:.1f} %')
print("top1 accuracy: ", (mean_accuracy / num_classes))