
# path_cifar10 = "/home/hojunson/Desktop/Datasets/CIFAR/cifar-10-python/"
#
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# batch_size = 4
#
# trainset = torchvision.datasets.CIFAR10(root=path_cifar10, train=True,
#                                         download=False, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root=path_cifar10, train=False,
#                                        download=False, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # functions to show an image
#
#
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# # get some random training images
# dataiter = iter(trainloader)
# images, labels = next(dataiter)
#
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


# path_cifar100 = "/home/hojunson/Desktop/Datasets/CIFAR/cifar-100-python/"
#
#
# CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
# CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
#
# transform_train = transforms.Compose([
#         #transforms.ToPILImage(),
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ToTensor(),
#         transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
#     ])
#
#
#
# #cifar100_training = CIFAR100Train(path, transform=transform_train)
# cifar100_training = torchvision.datasets.CIFAR100(root=path_cifar100, train=True, download=False, transform=transform_train)
# cifar100_training_loader = DataLoader(
#     cifar100_training, shuffle=True, num_workers=2, batch_size=256)
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
# ])
# # cifar100_test = CIFAR100Test(path, transform=transform_test)
# cifar100_test = torchvision.datasets.CIFAR100(root=path_cifar100, train=False, download=False, transform=transform_test)
# cifar100_test_loader = DataLoader(
#     cifar100_test, shuffle=True, num_workers=2, batch_size=256)


# path_cifar10_1 = "/home/hojunson/Desktop/Datasets/CIFAR/CIFAR-10.1-master/datasets"
#
# import utils_cifar10_1
# version = 'v6'
# images, labels = utils_cifar10_1.load_new_test_data(data_path=path_cifar10_1, version_string=version)
# num_images = images.shape[0]
#
# print('\nLoaded version "{}" of the CIFAR-10.1 dataset.'.format(version))
# print('There are {} images in the dataset.'.format(num_images))