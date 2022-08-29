
import numpy as np

import math
from tools import HEAT_SIZE,Heatmap_Show



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1]*(num_blocks - 1) # (stride == 2, num_blocks == 4 --> strides == [2, 1, 1, 1])

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion*channels

    layer = nn.Sequential(*blocks) # (*blocks: call with unpacked list entires as arguments)

    return layer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(x))) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn2(self.conv2(out)) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(x) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = F.relu(out) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(x))) # (shape: (batch_size, channels, h, w))
        out = F.relu(self.bn2(self.conv2(out))) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn3(self.conv3(out)) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(x) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = F.relu(out) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        return out
class ResNet_Bottleneck_OS16(nn.Module):
    def __init__(self, num_layers):
        super(ResNet_Bottleneck_OS16, self).__init__()

        if num_layers == 50:
            resnet = models.resnet50()
            # load pretrained model:
            # resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet50-19c8e357.pth"))
            # remove fully connected layer, avg pool and layer5:
            state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
            resnet.load_state_dict(state_dict, strict=False)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            print ("pretrained resnet, 50")
        elif num_layers == 101:
            resnet = models.resnet101()
            # load pretrained model:
            # resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet101-5d3b4d8f.pth"))
            state_dict = load_state_dict_from_url(model_urls['resnet101'], progress=True)
            resnet.load_state_dict(state_dict, strict=False)
            # remove fully connected layer, avg pool and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            print ("pretrained resnet, 101")
        elif num_layers == 152:
            resnet = models.resnet152()
            # load pretrained model:
            # resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet152-b121ed2d.pth"))
            state_dict = load_state_dict_from_url(model_urls['resnet152'], progress=True)
            resnet.load_state_dict(state_dict, strict=False)
            # remove fully connected layer, avg pool and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            print ("pretrained resnet, 152")
        else:
            raise Exception("num_layers must be in {50, 101, 152}!")

        self.layer5 = make_layer(Bottleneck, in_channels=4*256, channels=512, num_blocks=3, stride=1, dilation=2)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        # pass x through (parts of) the pretrained ResNet:
        c4 = self.resnet(x) # (shape: (batch_size, 4*256, h/16, w/16)) (it's called c4 since 16 == 2^4)

        output = self.layer5(c4) # (shape: (batch_size, 4*512, h/16, w/16))

        return output

class ASPP_Bottleneck(nn.Module):
    def __init__(self, num_classes=1,input_w=1024,input_h=800):
        super(ASPP_Bottleneck, self).__init__()
        self.input_w=input_w
        self.input_h=input_h

        self.conv_1x1_1 = nn.Conv2d(4*512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(4*512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(4*512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(4*512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(4*512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)


    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 4*512, h/16, w/16))
        # print(feature_map.shape)
        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        # print(out_img.shape)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
        # out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, h/16, w/16))
        out = F.upsample(out, size=(self.input_h,self.input_w ), mode="bilinear")

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes=256, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # print(avg_out.shape,max_out.shape)
        return self.sigmoid(out)



class Attention(nn.Module):
    def __init__(self,channel):
        super(Attention, self).__init__()
        self.fc1 = nn.Conv2d(channel, channel, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel, channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()



    def forward(self,x):
        b, c, h, w = x.shape
        x= self.fc2(self.relu1(self.fc1(x)))

        x = torch.sum(x, dim=1)
        x = x.reshape(b, 1, h, w)

        return self.sigmoid(x)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def ResNet50_OS16():
    return ResNet_Bottleneck_OS16(num_layers=50)

def ResNet101_OS16():
    return ResNet_Bottleneck_OS16(num_layers=101)
class SRDF(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super(SRDF, self).__init__()

        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))

        classes = heads['hm']
        REG = heads['REG']
        # self.resnet = ResNet50_OS16()  # NOTE! specify the type of ResNet here
        self.resnet = ResNet101_OS16()  # NOTE! specify the type of ResNet here
        self.aspp = ASPP_Bottleneck(num_classes=classes,input_h=HEAT_SIZE[0],input_w=HEAT_SIZE[1])

        self.ca = ChannelAttention(in_planes=256)
        self.sa = SpatialAttention()
        self.att = Attention(channel=256)

        self.branch1 = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                nn.ReLU(inplace=True)
                                )
        self.branch2 = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True)
                                     )
        self.branch3 = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True)
                                     )

        self.cls_head = nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
        self.reg_head = nn.Conv2d(head_conv, REG, kernel_size=3, padding=1, bias=True)
        self.theta_cls_head = nn.Conv2d(head_conv, 18, kernel_size=final_kernel, stride=1, padding=final_kernel // 2,bias=True)
        self.theta_reg_head = nn.Conv2d(head_conv, 1, kernel_size=3, padding=1, bias=True)

        for i in [self.branch1,self.branch2,self.branch3]:
            self.fill_fc_weights(i)
        prior = 0.01
        pa = (-math.log((1.0 - prior) / prior))
        self.cls_head.bias.data.fill_(-2.19)
        self.theta_cls_head.bias.data.fill_(pa)
        self.reg_head.bias.data.fill_(0)
        self.theta_reg_head.bias.data.fill_(0)


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        feature_map = self.resnet(x)
        c2_combine = self.aspp(feature_map)

        c2_combine = self.ca(c2_combine) * c2_combine
        att_head = self.att(c2_combine)
        c2_combine_att = att_head*c2_combine

        branch1 = self.branch1(c2_combine_att)
        branch2 = self.branch1(c2_combine)
        branch3 = self.branch1(c2_combine)

        cls_head = self.cls_head(branch1)
        reg_head = self.reg_head(branch2)
        theta_cls_head = self.theta_cls_head(branch3)
        theta_reg_head = self.theta_reg_head(branch3)

        dec_dict = {}
        dec_dict['hm'] = torch.sigmoid(cls_head)
        dec_dict['REG'] = reg_head
        dec_dict['theta_cls'] = torch.sigmoid(theta_cls_head)
        dec_dict['theta_reg'] = theta_reg_head
        dec_dict['att'] = att_head

        return dec_dict
if __name__ == '__main__':
    # ct_int = np.array([[1,1],[1,0]])
    # m = (ct_int[:,0] < 0) | (ct_int[:,1] < 0)
    # # mact1 =
    # # m = mact | mact1
    # print(len(ct_int[m]))
    heads = {'hm': 10,
             'REG':6,
             # 'offsets': 2,
             # 'regs': 2,
             # 'cls_theta': 1,
             # 'center_offsets': 1,
             'theta_cls':18,
             'theta_reg':1,
             # 'three':10
             }
    # print(data['input'].shape)
    # print(data['hm'].shape)
    # print(data['reg_mask'].shape)
    # print(data['ind'].shape)
    # print(data['size'].shape)
    # print(data['reg'].shape)
    # print(data['direction'].shape)
    down_ratio = 4
    model = SRDF(heads=heads,
                              pretrained=True,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              head_conv=256)
    #
    a = torch.randn((3,3,608,720))
    # for head in (model(a)):
    #     print(head)
    # print((model(a)))
    dictt = (model(a))
    print(dictt['hm'].shape)
    print(dictt['REG'].shape)
    # print(dictt['offsets'].shape)
    # print(dictt['regs'].shape)
    # # print(dictt['cls_theta'].shape)
    # print(dictt['center_offsets'].shape)
    print(dictt['theta_cls'].shape)
    print(dictt['theta_reg'].shape)