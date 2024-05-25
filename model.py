import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
from option import get_args
opt = get_args()

class My_CNN(nn.Module):
    def __init__(self):
        super(My_CNN, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(3, 16, (3, 3), 1, 1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(16))
        self.conv1_2 = nn.Sequential(nn.Conv2d(16, 32, (3, 3), 2, 1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(32))
        self.conv2_1 = nn.Sequential(nn.Conv2d(32, 32, (3, 3), 1, 1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(32))
        self.conv2_2 = nn.Sequential(nn.Conv2d(32, 64, (3, 3), 2, 1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(64))
        self.conv3_1 = nn.Sequential(nn.Conv2d(64, 64, (3, 3), 1, 1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(64))
        self.conv3_2 = nn.Sequential(nn.Conv2d(64, 128, (3, 3), 2, 1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(128))

        self.linear_1 = nn.Linear(28 * 28 * 128, 80)
        self.linear_2 = nn.Linear(80, 4)

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = x.view(in_size, -1)
        x = self.linear_1(x)
        out = self.linear_2(x)
        return out


def ResNet():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # for name, layer in model.named_children():  # Freeze only layer1
    #     if name == "layer1":
    #         for param in layer.parameters():
    #             param.requires_grad = False
    #
    # for param in model.parameters():    # Freeze all layers, lock all model parameters, and set all layers to untrained mode.
    #     param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    return model

if __name__ == '__main__':
    model = ResNet().to(opt.device)

