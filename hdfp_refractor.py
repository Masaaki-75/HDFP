# coding=utf-8 #

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
HDFPNet - Basic Implementation
Paper : Holistic and Deep Feature Pyramids for Saliency Detection
"""


class upsample(nn.Module):
    def __init__(self, factor):
        super(upsample, self).__init__()
        self.factor = factor

    def forward(self, x):
        y = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=True)
        return y


class avg_pool(nn.Module):
    def __init__(self, s):
        super(avg_pool, self).__init__()
        self.s = s

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=[self.s, self.s], stride=[self.s, self.s], padding=0)


class batch_activ_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, drop_rate=0):
        super(batch_activ_conv, self).__init__()
        self.drop_rate = drop_rate

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation, bias=True,
                              stride=1, padding=(dilation * (kernel_size - 1)) // 2)
        self.dropout = nn.Dropout2d(p=drop_rate)

    def forward(self, x):
        y = self.relu(self.bn(x))
        # y = F.pad(y, pad=[0, 1, 1, 0], mode='constant', value=0)
        y = self.conv(y)
        return self.dropout(y)


class block(nn.Module):
    def __init__(self, in_channels, growth, n_layers=12, dilation=1, drop_rate=0):
        super(block, self).__init__()
        self.dense_layers = self._make_layers(in_channels, growth, n_layers, 3, dilation, drop_rate)

    def _make_layers(self, in_channels, growth, n_layers, kernel_size, dilation, drop_rate):
        layers = []
        n_channels = in_channels
        for _ in range(n_layers):
            layers.append(batch_activ_conv(n_channels, growth, kernel_size,
                                           dilation=dilation, drop_rate=drop_rate))
            n_channels += growth
        return nn.Sequential(*layers)

    def forward(self, x):
        y = x
        for _, layer in self.dense_layers.named_children():
            tmp = layer(y)
            y = torch.cat((y, tmp), dim=1)
        return y


class pyramid_pooling_64(nn.Module):
    def __init__(self, in_channels, pyramid_channels, scale_rates=None):
        super(pyramid_pooling_64, self).__init__()
        scale_rates = [64, 32, 16, 8] if scale_rates is None else scale_rates
        self.layer1 = nn.Sequential(avg_pool(scale_rates[0]),
                                    nn.Conv2d(in_channels, pyramid_channels, 1, padding=0),
                                    upsample(scale_rates[0]))
        self.layer2 = nn.Sequential(avg_pool(scale_rates[1]),
                                    nn.Conv2d(in_channels, pyramid_channels, 1, padding=0),
                                    upsample(scale_rates[1]))
        self.layer3 = nn.Sequential(avg_pool(scale_rates[2]),
                                    nn.Conv2d(in_channels, pyramid_channels, 1, padding=0),
                                    upsample(scale_rates[2]))
        self.layer4 = nn.Sequential(avg_pool(scale_rates[3]),
                                    nn.Conv2d(in_channels, pyramid_channels, 1, padding=0),
                                    upsample(scale_rates[3]))

    def forward(self, x):
        pyramid = torch.cat((self.layer1(x), self.layer2(x), self.layer3(x), self.layer4(x)), dim=1)
        return pyramid


class HDFPNet(nn.Module):
    def __init__(self, in_channels, n_classes, drop_rate, n_layers=12, softmax=False, deep_supervision=True):
        super(HDFPNet, self).__init__()
        self.softmax = softmax
        self.deep_supervision = deep_supervision

        growth = 12
        fuse_channels = 16
        reduction = 0.5  # reduction rate in BatchActivConv module
        dilated_rates = [1, 1, 2, 4, 8]
        pyramid_channels = 1  # Number of channels in each layer of PyramidPooling module

        self.conv_in = nn.Conv2d(in_channels, fuse_channels, 3, stride=1, padding=1, bias=True)
        # Level 1 (down): 256 x 256, d = 1
        self.block1 = block(fuse_channels, growth, n_layers=n_layers, dilation=dilated_rates[0], drop_rate=drop_rate)
        block1_channels = fuse_channels + n_layers * growth  # Number of output channels of Block 1, i.e. 160
        down1_channels = int(block1_channels * reduction)  # Number of output channels of Level 1 (down), i.e. 80
        self.conv1 = nn.Conv2d(block1_channels, fuse_channels, 3, stride=1, padding=1, bias=True)
        self.bac1 = batch_activ_conv(block1_channels, down1_channels, 1, drop_rate=drop_rate)
        self.avg1 = avg_pool(s=2)

        # Level 2 (down): 128 x 128, d = 1
        self.block2 = block(down1_channels, growth, n_layers=n_layers, dilation=dilated_rates[1], drop_rate=drop_rate)
        block2_channels = down1_channels + n_layers * growth  # 224
        down2_channels = int(block2_channels * reduction)  # 112
        self.conv2 = nn.Conv2d(block2_channels, fuse_channels, 3, stride=1, padding=1, bias=True)
        self.bac2 = batch_activ_conv(block2_channels, down2_channels, 1, drop_rate=drop_rate)
        self.avg2 = avg_pool(s=2)

        # Level 3 (down): 64 x 64, d = 2
        self.block3 = block(down2_channels, growth, n_layers=n_layers, dilation=dilated_rates[2], drop_rate=drop_rate)
        block3_channels = down2_channels + n_layers * growth  # 256
        down3_channels = int(block3_channels * reduction)  # 128
        self.conv3 = nn.Conv2d(block3_channels, fuse_channels, 3, stride=1, padding=1, bias=True)
        self.bac3 = batch_activ_conv(block3_channels, down3_channels, 1, drop_rate=drop_rate)

        # Level 4 (down): 64 x 64, d = 4
        self.block4 = block(down3_channels, growth, n_layers=n_layers, dilation=dilated_rates[3], drop_rate=drop_rate)
        block4_channels = down3_channels + n_layers * growth  # 272
        down4_channels = int(block4_channels * reduction)  # 136
        self.conv4 = nn.Conv2d(block4_channels, fuse_channels, 3, stride=1, padding=1, bias=True)
        self.bac4 = batch_activ_conv(block4_channels, down4_channels, 1, drop_rate=drop_rate)

        # Level 5 (down): 64 x 64, d = 8
        self.block5 = block(down4_channels, growth, n_layers=n_layers, dilation=dilated_rates[4], drop_rate=drop_rate)
        block5_channels = down4_channels + n_layers * growth  # 280
        down5_channels = fuse_channels  # 16
        self.conv5 = nn.Conv2d(block5_channels, fuse_channels, 3, stride=1, padding=1, bias=True)

        # Level 5 (up): logits_64_3
        self.ppm64 = pyramid_pooling_64(down5_channels, pyramid_channels=pyramid_channels)
        up5_channels = fuse_channels + 4 * pyramid_channels  # Number of output channels of Level 5 (up), i.e. 20
        self.bac_deep5 = batch_activ_conv(up5_channels, n_classes, 3)

        # Level 4 (up): logits_64_2
        up4_channels = fuse_channels + up5_channels  # 16 + 20 = 36
        self.bac_deep4 = batch_activ_conv(up4_channels, n_classes, 3)

        # Level 3 (up): logits_64_1
        up3_channels = fuse_channels + up4_channels  # 16 + 36 = 52
        self.bac_deep3 = batch_activ_conv(up3_channels, n_classes, 3)

        # Level 2 (up): upsample to 128
        self.upsample128 = upsample(factor=2)
        up2_channels = fuse_channels + up3_channels  # 16 + 52 = 68
        self.bac_deep2 = batch_activ_conv(up2_channels, n_classes, 3)

        # Level 1 (up): upsample to 256
        self.upsample256 = upsample(factor=2)
        up1_channels = fuse_channels + up2_channels  # 16 + 68 = 84
        self.bac_deep1 = batch_activ_conv(up1_channels, n_classes, 3)
        self.upsample_2x = upsample(factor=2)
        self.upsample_4x = upsample(factor=4)

        # final convolution
        self.conv_out1 = nn.Conv2d(5 * n_classes, n_classes, 3, stride=1, padding=1, bias=True)
        self.conv_out2 = nn.Conv2d(n_classes, n_classes, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        current = self.conv_in(x)

        # Level 1 (down): 256 x 256, d = 1
        current = self.block1(current)  # 160, 256, 256
        scale_256 = self.conv1(current)  # 16, 256, 256
        current = self.bac1(current)  # 80, 256, 256
        current = self.avg1(current)  # 80, 128, 128

        # Level 2 (down): 128 x 128, d = 1
        current = self.block2(current)  # 224, 128, 128
        scale_128 = self.conv2(current)  # 16, 128, 128
        current = self.bac2(current)  # 112, 128, 128
        current = self.avg2(current)  # 112, 64, 64

        # Level 3 (down): 64 x 64 ,d = 2
        current = self.block3(current)  # 256, 64, 64
        scale_64_1 = self.conv3(current)  # 16, 64, 64
        current = self.bac3(current)  # 128, 64, 64

        # Level 4 (down): 64 x 64, d = 4
        current = self.block4(current)  # 272, 64, 64
        scale_64_2 = self.conv4(current)  # 16, 64, 64
        current = self.bac4(current)  # 136, 64, 64

        # Level 5 (down): 64 * 64, d = 8
        current = self.block5(current)  # 280, 64, 64
        scale_64_3 = self.conv5(current)  # 16, 64, 64

        # Level 5 (up): 64_3 Map
        ppm_64_3 = self.ppm64(scale_64_3)  # 4, 64, 64
        concat_64_3 = torch.cat((scale_64_3, ppm_64_3), 1)  # 20, 64, 64
        logits_64_3 = self.bac_deep5(concat_64_3)  # 3, 64, 64

        # Level 4 (up): 64_2 Map
        concat_64_2 = torch.cat((scale_64_2, concat_64_3), 1)  # 36, 64, 64
        logits_64_2 = self.bac_deep4(concat_64_2)  # 3, 64, 64

        # Level 3 (up): 64_1 Map
        concat_64_1 = torch.cat((scale_64_1, concat_64_2), 1)  # 52, 128, 128
        logits_64_1 = self.bac_deep3(concat_64_1)  # 3, 64, 64

        # Level 2 (up): recovery 128
        concat_64_1_up = self.upsample128(concat_64_1)  # 52, 128, 128
        concat_128 = torch.cat((scale_128, concat_64_1_up), 1)  # 68, 128, 128
        logits_128 = self.bac_deep2(concat_128)  # 3, 128, 128

        # Level 1 (up): recovery 256
        logits_128_up = self.upsample256(concat_128)  # 68, 256, 256
        concat_256 = torch.cat((scale_256, logits_128_up), 1)  # 84, 256, 256
        logits_256 = self.bac_deep1(concat_256)  # 3, 256, 256

        # All upsampled to 256
        logits_64_3_up_256 = self.upsample_4x(logits_64_3)
        logits_64_2_up_256 = self.upsample_4x(logits_64_2)
        logits_64_1_up_256 = self.upsample_4x(logits_64_1)
        logits_128_up_256 = self.upsample_2x(logits_128)

        logits_64_3_prob = F.softmax(logits_64_3_up_256, dim=1) if self.softmax else torch.sigmoid(logits_64_3_up_256)
        logits_64_2_prob = F.softmax(logits_64_2_up_256, dim=1) if self.softmax else torch.sigmoid(logits_64_2_up_256)
        logits_64_1_prob = F.softmax(logits_64_1_up_256, dim=1) if self.softmax else torch.sigmoid(logits_64_1_up_256)
        logits_128_prob = F.softmax(logits_128_up_256, dim=1) if self.softmax else torch.sigmoid(logits_128_up_256)
        logits_256_prob = F.softmax(logits_256, dim=1) if self.softmax else torch.sigmoid(logits_256)
        logits = self.conv_out1(torch.cat((logits_64_3_up_256, logits_64_2_up_256, logits_64_1_up_256,
                                           logits_128_up_256, logits_256), dim=1))
        logits = self.conv_out2(logits)
        yp = F.softmax(logits, dim=1) if self.softmax else torch.sigmoid(logits)

        if self.deep_supervision:
            return yp, logits_64_3_prob, logits_64_2_prob, logits_64_1_prob, logits_128_prob, logits_256_prob
        else:
            return yp
