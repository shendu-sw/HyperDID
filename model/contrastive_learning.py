import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn import init
from model.TPPI.models.utils import *
#import torchsnooper

import math
import os
import datetime
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


#@torchsnooper.snoop()
class ContrastiveLearn(nn.Module):
    """
    Based on paper:HybridSN: Exploring 3-D-2-D CNN Feature Hierarchy for Hyperspectral Image Classification. IEEE Geoscience and Remote Sensing Letters
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, band, num_classes, patch_size=5): # band  spectral channels  # num_classes  number of classes
        super(ContrastiveLearn, self).__init__()
        
        self.input_channels = band
        self.patch_size = patch_size

        self.FE = nn.Sequential(
            nn.Conv2d(in_channels=band, out_channels=30, kernel_size=(1, 1), stride=(1, 1)),
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

        self.feature_size = self._get_final_flattened_size()
        self.FC1 = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )

        self.FC1_2 = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )

        self.FC2_2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )

        self.classifier = nn.Linear(128, num_classes)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels,
                             self.patch_size, self.patch_size))
            '''x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)'''
            #x = rearrange(x, 'b h d n -> b n h d')
            fe = self.FE(x)
            fe = torch.unsqueeze(fe, 1)
            conv1 = self.conv1(fe)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv3 = torch.reshape(conv3, (
            conv3.shape[0], -1, conv3.shape[3], conv3.shape[4]))
            x = self.conv4(conv3)
            #conv4 = torch.reshape(conv4, (conv4.shape[0], -1))
            _, c, w, h = x.size()
        return c * w * h

    def forward(self, x):
        #x = torch.squeeze(x,1)
        x = rearrange(x, 'b h d n -> b n h d')
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

        fc1_2 = self.FC1_2(conv4)
        fc2_2 = self.FC2_2(fc1_2)

        fc = fc2 * fc2_2

        out = self.classifier(fc)
        return fc2, fc2_2, out#out


class GRU(nn.Module):
    def __init__(self, num_classes, patches):
        super(GRU, self).__init__()
        HiddenSize = 512   # we choose 128, 192 may achieve a little better results
        LstmLayers = 1
        self.GRU = nn.GRU(  # if use nn.RNN(), it hardly learns
            input_size=patches*patches,
            hidden_size=HiddenSize,  # rnn hidden unit
            num_layers=LstmLayers,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # dropout=0.5
        )
        # self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(HiddenSize, 128)
        
        self.fc1 = nn.Linear(HiddenSize, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        x = rearrange(x, 'b h d n -> b n (h d)')
        #print('x:', x.shape)
        r_out, h_n = self.GRU(x, None)
        # r_out = self.dropout(r_out)
        

        feature1 = self.fc(r_out[:, -1, :])
        feature2 = self.fc1(r_out[:, -1, :])
        #out = self.out(r_out[:, -1, :])
        feature = feature1 * feature2

        x = self.out(feature)

        return feature1, feature2, x

        #return out


class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1)
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0)
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))

        #self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, 128)
        
        self.fc1 = nn.Linear(self.features_size, 128)
        # self.classifier1 = nn.Linear(100, n_classes)

        self.classifier = nn.Linear(128, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = rearrange(x, 'b h d n -> b n h d')
        x=x.unsqueeze(1)
        #print('x:', x.shape)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        #x = self.dropout(x)
        feature1 = self.fc(x)
        feature2 = self.fc1(x)

        feature = feature1 * feature2

        x = self.classifier(feature)

        return feature1, feature2, x


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

        self.fc = nn.Linear(98*4, 128)
        self.fc1 = nn.Linear(98*4, 128)

        self.classifier = nn.Linear(128, get_class_num(dataset))

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

        feature =  self.fc(avg)
        feature1 = self.fc1(avg)

        avg = feature * feature1

        out = self.classifier(avg)
        return feature, feature1, out


class PMLP(nn.Module):
    
    def __init__(self, layers=[4, 64, 128], num_classes=5):
        super(PMLP, self).__init__()
        linear_layers = []
        for i in range(len(layers) - 2):
            linear_layers.append(nn.Linear(layers[i], layers[i + 1]))
            linear_layers.append(nn.GELU())
        linear_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers = nn.Sequential(*linear_layers)
        self.outdim = layers[-1]

        self.classifier = nn.Linear(self.outdim, num_classes)

    def forward(self, x):
        out = self.layers(x)
        #cl = self.classifier(out)
        return out


class MLP(nn.Module):
    
    def __init__(self, layers=[4, 64, 128], num_classes=5):
        super(MLP, self).__init__()
        linear_layers = []
        for i in range(len(layers) - 2):
            linear_layers.append(nn.Linear(layers[i], layers[i + 1]))
            linear_layers.append(nn.GELU())
        linear_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers = nn.Sequential(*linear_layers)
        self.outdim = layers[-1]

        self.classifier = nn.Linear(self.outdim, num_classes)

    def forward(self, x):
        out = self.layers(x)
        #cl = self.classifier(out)
        return out


if __name__ == "__main__":
    """
    open torchsnooper-->test the shape change of each model
    """
    model = MLP()
    a = np.random.random((1, 200, 1, 1))  # NCL
    a = torch.from_numpy(a).float()
    b = model(a)

    model = SSRN('SV')
    a = np.random.random((2, 204, 5, 5))  # NCHW
    a = torch.from_numpy(a).float()
    b = model(a)
    print(b)

