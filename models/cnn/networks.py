import torch
import torch.nn as nn


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, padding=1, normalize=True):
        super(_ConvBlock, self).__init__()
        self.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding))
        self.append(nn.ReLU(inplace=True))

        if normalize:
            self.insert(1, nn.BatchNorm2d(out_channels))


class LowLevelFeaturesNetwork(nn.Module):
    def __init__(self):
        super(LowLevelFeaturesNetwork, self).__init__()
        self.layers = nn.Sequential(
            _ConvBlock(1, 64, stride=2),
            _ConvBlock(64, 128, stride=1),

            _ConvBlock(128, 128, stride=2),
            _ConvBlock(128, 256, stride=1),

            _ConvBlock(256, 256, stride=2),
            _ConvBlock(256, 512, stride=1)
        )

    def forward(self, x):
        return self.layers(x)


class MidLevelFeaturesNetwork(nn.Module):
    def __init__(self):
        super(MidLevelFeaturesNetwork, self).__init__()
        self.layers = nn.Sequential(
            _ConvBlock(512, 512, stride=1),
            _ConvBlock(512, 256, stride=1)
        )

    def forward(self, x):
        return self.layers(x)


class GlobalFeaturesNetwork(nn.Module):
    def __init__(self, num_classes=1000):
        super(GlobalFeaturesNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            _ConvBlock(512, 512, stride=2),
            _ConvBlock(512, 512, stride=1),

            _ConvBlock(512, 512, stride=2),
            _ConvBlock(512, 512, stride=1)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * 512, 1024),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.fc_classification = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)

        x = self.fc1(x)
        x = self.fc2(x)

        return self.fc3(x), self.fc_classification(x)


class FusionBlock(nn.Module):
    def __init__(self):
        super(FusionBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2 * 256, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_mid, x_global):
        B, C, H, W = x_mid.shape
        x_global_expand = x_global.view(B, C, 1, 1).expand(-1, -1, H, W)
        x_concat = torch.cat((x_global_expand, x_mid), dim=1)

        return self.layers(x_concat)


class ColorizationNetwork(nn.Module):
    def __init__(self):
        super(ColorizationNetwork, self).__init__()
        self.layers = nn.Sequential(
            _ConvBlock(256, 128, stride=1),

            nn.Upsample(scale_factor=2, mode='nearest'),
            _ConvBlock(128, 64, stride=1),
            _ConvBlock(64, 64, stride=1),

            nn.Upsample(scale_factor=2, mode='nearest'),
            _ConvBlock(64, 32, stride=1),
            nn.Conv2d(32, 2, stride=1, padding=1, kernel_size=3),
            nn.Sigmoid(),

            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        return self.layers(x)
