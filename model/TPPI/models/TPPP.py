import torch
import torch.nn as nn
import numpy as np
#import torchsnooper
from model.TPPI.models.utils import *
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


#@torchsnooper.snoop()
class HybridSN(nn.Module):
    """
    Based on paper:HybridSN: Exploring 3-D-2-D CNN Feature Hierarchy for Hyperspectral Image Classification. IEEE Geoscience and Remote Sensing Letters
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset):
        super(HybridSN, self).__init__()
        self.dataset = dataset
        self.FE = nn.Sequential(
            nn.Conv2d(in_channels=get_in_channel(dataset), out_channels=30, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(30),
        )
        self.conv1 = nn.Sequential(
            # Notice:cause input shape is [N,C,D,H,W]，kernel_size here should be (D,H,W)
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=576, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.FC1 = nn.Sequential(
            nn.Linear(get_fc_in(dataset, 'HybridSN'), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.classifier = nn.Linear(128, get_class_num(dataset))

    def forward(self, x):
        x = torch.squeeze(x,1)
        fe = self.FE(x)
        fe = torch.unsqueeze(fe, 1)
        conv1 = self.conv1(fe)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv3 = torch.reshape(conv3, (
        conv3.shape[0], -1, conv3.shape[3], conv3.shape[4]))
        conv4 = self.conv4(conv3)
        conv4 = torch.reshape(conv4, (conv4.shape[0], -1))
        fc1 = self.FC1(conv4)
        fc2 = self.FC2(fc1)
        out = self.classifier(fc2)
        return out


# @torchsnooper.snoop()
class pResNet(nn.Module):
    """
    Based on paper:Paoletti. Deep pyramidal residual networks for spectral-spatial hyperspectral image classification. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    In source code, each layer have 3 bottlenecks, i change to 2 bottlenecks each layer, but still with 3 layer
    """
    def __init__(self, dataset):
        super(pResNet, self).__init__()
        self.dataset = dataset
        self.in_planes = get_in_planes(dataset)
        self.FE = nn.Sequential(
            nn.Conv2d(get_in_channel(dataset), self.in_planes, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.in_planes),
        )
        self.layer1 = nn.Sequential(
            Bottleneck_TPPP(self.in_planes, 43),
            Bottleneck_TPPP(43*4, 54),
        )
        self.reduce1 = Bottleneck_TPPP(54 * 4, 54, stride=2, downsample=nn.AvgPool2d((2, 2), (2, 2)))
        self.layer2 = nn.Sequential(
            Bottleneck_TPPP(54*4, 65),
            Bottleneck_TPPP(65*4, 76),
        )
        self.reduce2 = Bottleneck_TPPP(76*4, 76, stride=2, downsample=nn.AvgPool2d((2, 2), (2, 2)))
        self.layer3 = nn.Sequential(
            Bottleneck_TPPP(76*4, 87),
            Bottleneck_TPPP(87*4, 98),
        )
        self.avgpool = nn.AvgPool2d(get_avgpoosize(dataset))
        self.classifier = nn.Linear(98*4, get_class_num(dataset))

    def forward(self, x):
        x = rearrange(x, 'b h d n -> b n h d')
        #x = torch.squeeze(x,1)
        FE = self.FE(x)  # 降维
        layer1 = self.layer1(FE)
        reduce1 = self.reduce1(layer1)
        layer2 = self.layer2(reduce1)
        reduce2 = self.reduce2(layer2)
        layer3 = self.layer3(reduce2)
        avg = self.avgpool(layer3)
        avg = avg.view(avg.size(0), -1)
        out = self.classifier(avg)
        return out


if __name__ == "__main__":
    """
    open torchsnooper-->test the shape change of each model
    """
    model = CNN_1D('IP')
    a = np.random.random((1, 200, 1, 1))  # NCL
    a = torch.from_numpy(a).float()
    b = model(a)

    # model = CNN_2D('IP')
    # a = np.random.random((2, 200, 5, 5))  # NCHW
    # a = torch.from_numpy(a).float()
    # b = model(a)

    # model = CNN_3D('SV')
    # a = np.random.random((2, 204, 5, 5))  # NCHW
    # a = torch.from_numpy(a).float()
    # b = model(a)

    # model = HybridSN('IP')
    # a = np.random.random((2, 200, 5, 5))  # NCHW
    # a = torch.from_numpy(a).float()
    # b = model(a)

    # model = SSAN('IP')
    # a = np.random.random((2, 200, 5, 5))  # NCHW
    # a = torch.from_numpy(a).float()
    # b = model(a)

    # model = pResNet('IP')
    # a = np.random.random((2, 200, 5, 5))  # NCHW
    # a = torch.from_numpy(a).float()
    # a = a.cuda()
    # model = model.cuda()
    # b = model(a)

    model = SSRN('SV')
    a = np.random.random((2, 204, 5, 5))  # NCHW
    a = torch.from_numpy(a).float()
    b = model(a)
    print(b)

