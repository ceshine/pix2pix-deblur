import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset


class SimpleBlur(object):

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.filter = Variable(torch.FloatTensor(
            3, 3, kernel_size, kernel_size).fill_(0), requires_grad=False)
        for i in range(3):
            self.filter[i, i, :, :] = 1 / kernel_size / kernel_size

    def __call__(self, tensor):
        return F.conv2d(Variable(tensor, requires_grad=False), self.filter).data


class GaussianBlur(SimpleBlur):

    def __init__(self, kernel_size, std):
        self.kernel_size = kernel_size
        center = (kernel_size + 1) / 2
        variance = std ** 2
        tmp_kernel = np.zeros((kernel_size, kernel_size))
        # TODO: make efficient
        for i in range(kernel_size):
            for j in range(kernel_size):
                tmp_kernel[i, j] = (
                    (i + 1 - center)**2 + (j + 1 - center)**2
                ) / variance * -1
        tmp_kernel = np.exp(tmp_kernel)
        tmp_kernel = tmp_kernel / np.sum(tmp_kernel)  # make sum to one
        print("Kernel:", tmp_kernel)
        tmp_filter = np.zeros((3, 3, kernel_size, kernel_size))
        for i in range(3):
            tmp_filter[i, i, :, :] = tmp_kernel
        self.filter = Variable(
            torch.from_numpy(tmp_filter), requires_grad=False
        ).float()


DEFAULT_TRANSFORMS = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomSizedCrop(132),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])


class BlurDataset(Dataset):
    def __init__(self, path, blur, transforms=DEFAULT_TRANSFORMS):
        self.img_folder = ImageFolder(path, transforms)
        self.shrink_size = blur.kernel_size // 2
        self.blur = blur

    def __getitem__(self, idx):
        X, _ = self.img_folder[idx]
        return (
            self.blur(X.unsqueeze(0)).squeeze(0),
            X[:, self.shrink_size:-self.shrink_size,
                self.shrink_size:-self.shrink_size]
        )

    def __len__(self):
        return len(self.img_folder)


class InverseNormalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


INVERSE_NORMALIZE = InverseNormalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
