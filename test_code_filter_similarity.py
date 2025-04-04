from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
import torch.nn as nn
from datasets.Imagenet import Imagenet_datasets
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from Ts_ss_utils import vector_similarity_vectorized as tsss

# New weights with accuracy 80.858%
# model2 = resnet50(weights=None)
#
# model2.fc = nn.Linear(2048, 10)
#
#
# print(model2)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

training_data = Imagenet_datasets(is_train=False, target_img_w=224, target_img_h=224)

training_data_loader = DataLoader(training_data, batch_size=1,
                                 shuffle=True, num_workers=1)

test_data_imagenet = Imagenet_datasets(is_train=False, target_img_w=224, target_img_h=224)
test_data_loader = DataLoader(test_data_imagenet, batch_size=1,
                                 shuffle=False, num_workers=1)

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

num_classes = 1000
correct_pred = {str(idx): 0 for idx in range(0, num_classes)}
total_pred = {str(idx): 0 for idx in range(0, num_classes)}
device = torch.device('cuda:0')
model.to(device)

activations = dict()
patches = dict()


def conv1_activations_hook(module, input, output):
    global activations, patches

    inp_tensor = input[0].detach()

    # unfold the input tensor to extract 3x3 patches
    patches['conv1'] = inp_tensor.unfold(2, 3, 1).unfold(3, 3, 1)

    activations['conv1'] = output
    return output

def layer_1_1_conv2_activations_hook(module, input, output):
    global activations, patches

    inp_tensor = input[0].detach()

    # unfold the input tensor to extract 3x3 patches
    patches['1_1'] = inp_tensor.unfold(2, 3, 1).unfold(3, 3, 1)

    activations['1_1'] = output
    return output

def layer_2_1_conv2_activations_hook(module, input, output):
    global activations

    inp_tensor = input[0].detach()

    # unfold the input tensor to extract 3x3 patches
    patches['2_1'] = inp_tensor.unfold(2, 3, 1).unfold(3, 3, 1)

    activations['2_1'] = output
    return output

def layer_3_1_conv2_activations_hook(module, input, output):
    global activations

    inp_tensor = input[0].detach()

    # unfold the input tensor to extract 3x3 patches
    patches['3_1'] = inp_tensor.unfold(2, 3, 1).unfold(3, 3, 1)

    activations['3_1'] = output
    return output

def layer_4_1_conv2_activations_hook(module, input, output):
    global activations

    inp_tensor = input[0].detach()

    # unfold the input tensor to extract 3x3 patches
    patches['4_1'] = inp_tensor.unfold(2, 3, 1).unfold(3, 3, 1)

    activations['4_1'] = output
    return output

for (name, module) in model.named_modules():
    print(name)
    if name == "conv1":
        print(f"Layer {name}  found in Model! Register forward hook")
        module.register_forward_hook(conv1_activations_hook)

    if name == "layer1.1.conv2":
        print(f"Layer {name}  found in Model! Register forward hook")
        module.register_forward_hook(layer_1_1_conv2_activations_hook)

    if name == "layer2.1.conv2":
        print(f"Layer {name}  found in Model! Register forward hook")
        module.register_forward_hook(layer_2_1_conv2_activations_hook)

    if name == "layer3.1.conv2":
        print(f"Layer {name}  found in Model! Register forward hook")
        module.register_forward_hook(layer_3_1_conv2_activations_hook)

    if name == "layer4.1.conv2":
        print(f"Layer {name}  found in Model! Register forward hook")
        module.register_forward_hook(layer_4_1_conv2_activations_hook)

target_layer = "2_1"
similarity = tsss.TS_SS()


for data in training_data_loader:

    images = data["images"].to(device)
    outputs = model(images)

    print(activations[target_layer].shape)

    # Access a particular spatial location (e.g., location at H_out=10, W_out=10)
    h_loc, w_loc = 0, 0  # Choose a valid spatial location
    single_patch1 = patches[target_layer][0, :, h_loc, w_loc, :, :]  # Shape: [C, 3, 3]

    # filter1 = torch.histogram(single_patch1.cpu(), bins= 10)

    h_loc, w_loc = 2, 2  # Choose a valid spatial location
    single_patch2 = patches[target_layer][0, :, h_loc, w_loc, :, :]  # Shape: [C, 3, 3]

    filter2 = torch.histogram(single_patch2.cpu(), bins= 10)
    temp1 = single_patch1.reshape(1,-1)
    temp2 = single_patch2.reshape(1,-1)
    cossim = 1 - F.cosine_similarity(temp1, temp2)

    ''' draw similarity maps per each location '''
    B, C, H, W = activations[target_layer].shape

    maps = np.zeros((H, W))

    for h in range(1,H-2):
        for w in range(1,W-2):
            single_patch1 = patches[target_layer][0, :, h, w, :, :]  # Shape: [C, 3, 3]
            temp1 = single_patch1.reshape(1, -1)

            sim_list = []
            for h2 in range(h-1, h+1):
                for w2 in range(w-1, w+1):
                    if h2 == h and w2 == w:
                        continue

                    if h2 <0 or w2 < 0: continue

                    if h2 >=H or w2 >= W: continue

                    single_patch2 = patches[target_layer][0, :, h2, w2, :, :]  # Shape: [C, 3, 3]

                    temp2 = single_patch2.reshape(1, -1)
                    # sim = 1 - F.cosine_similarity(temp1, temp2)
                    # sim_list.append(sim.cpu().numpy())

                    sim = similarity(torch.squeeze(temp1,dim=0) .cpu().numpy(), torch.squeeze(temp2,dim=0).cpu().numpy())
                    sim_list.append(sim)

            maps[h, w] = np.mean(sim_list)

    print(f"h: {h}, w: {w}")

    images = images.cpu()
    images = images * np.expand_dims(
        np.expand_dims(np.expand_dims(torch.as_tensor(np.array([0.229, 0.224, 0.225])), axis=0), axis=-1), axis=-1)
    images = images + np.expand_dims(
        np.expand_dims(np.expand_dims(torch.as_tensor(np.array([0.485, 0.456, 0.406])), axis=0), axis=-1), axis=-1)
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


    imshow(torchvision.utils.make_grid(images))

    plt.imshow(maps)
    plt.show()

    ''' based on the similarity values draw border lines?'''
    # print(single_patch)
