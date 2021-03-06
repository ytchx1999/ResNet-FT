import torch
import torchvision
from torchsummary import summary
import torch.nn as nn
import os
import wget


class ResNet(nn.Module):
    def __init__(self, pretain_dir, num_classes=1000):
        super(ResNet, self).__init__()
        self.pretrain_dir = pretain_dir
        self.num_classes = num_classes
        self.resnet_18 = torchvision.models.resnet18()

        if not os.path.exists(self.pretrain_dir):
            # download
            # https://download.pytorch.org/models/resnet18-5c106cde.pth
            url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
            wget.download(url, "./resnet18-5c106cde.pth")

        state_dict_load = torch.load(self.pretrain_dir, map_location='cpu')
        self.resnet_18.load_state_dict(state_dict_load)

        num_features = self.resnet_18.fc.in_features
        self.resnet_18.fc = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        x = self.resnet_18(x)
        return x


if __name__ == '__main__':
    resnet_18 = ResNet('resnet18-5c106cde.pth', num_classes=2)
    summary(resnet_18, input_size=(3, 32, 32))
    # print(os.path.exists('./resnet18-5c106cde.pth'))
