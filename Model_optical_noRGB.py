import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from network import *
# from PWCNet import PWCNetwork
import math
import numpy as np
from VideoDataSet import show_flow_hsv
import torch.nn.functional as F
from pwclite import PWCLite

import os
from torch.cuda.amp import autocast as autocast

from torchsummaryX import summary

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
class SpatialCNN(nn.Module):
    def __init__(self, arch, num_classes):
        super(SpatialCNN, self).__init__()
        self.num_classes = num_classes

        if arch.startswith('resnet18'):
            self.cnn = resnet18(pretrained=True, channel=3, num_classes=num_classes)

    def forward(self, inputs):
        outputs = self.cnn(inputs)
        return outputs

class MotionCNN(nn.Module):
    def __init__(self, arch, channels, num_classes):
        super(MotionCNN, self).__init__()
        self.num_classes = num_classes
        self.channels = channels

        if arch.startswith('resnet18'):
            self.cnn = resnet18(pretrained=True, channel=channels, num_classes=num_classes)

    def forward(self, inputs):
        outputs = self.cnn(inputs)
        return outputs

class CNN_LSTM(nn.Module):
    def __init__(self, arch, num_classes, lstm_layers, hidden_size):
        super(CNN_LSTM, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

        if arch.startswith('resnet18'):
            original_model = models.__dict__['resnet18'](pretrained=True).cuda()
            self.cnn = nn.Sequential(*list(original_model.children())[:-1])
            for i, param in enumerate(self.cnn.parameters()):
                param.requires_grad = True
        self.rnn = nn.LSTM(input_size=512, hidden_size=self.hidden_size, num_layers=self.lstm_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

        for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
             elif isinstance(m, nn.LSTM):
                nn.init.orthogonal_(m.weight_ih_l0)
                nn.init.constant_(m.bias_ih_l0, 0)

    def forward(self, inputs):
        # hidden = None
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        input = torch.zeros(batch_size,seq_len,self.rnn.input_size).cuda()
        for t in range(seq_len): #seq_len
            x = self.cnn(inputs[:, t, :, :, :])
            input[:,t,:] = x.squeeze() #B*512
        out, _ = self.rnn(input, None)
        x = self.fc(out[:, -1, :])
        return x

class CNN_LSTM_rgb(nn.Module):
    def __init__(self, arch, num_classes, lstm_layers, hidden_size):
        super(CNN_LSTM_rgb, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

        if arch.startswith('resnet18'):
            original_model = models.__dict__['resnet18'](pretrained=True).cuda()
            self.cnn = nn.Sequential(*list(original_model.children())[:-1])
            for i, param in enumerate(self.cnn.parameters()):
                param.requires_grad = True
        self.rnn = nn.LSTM(input_size=512, hidden_size=self.hidden_size, num_layers=self.lstm_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

        for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
             elif isinstance(m, nn.LSTM):
                nn.init.orthogonal_(m.weight_ih_l0)
                nn.init.constant_(m.bias_ih_l0, 0)

    def forward(self, inputs):
        # hidden = None
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        input = torch.zeros(batch_size,seq_len,self.rnn.input_size).cuda()
        for t in range(seq_len): #seq_len
            x = self.cnn(inputs[:, t, :, :, :])
            input[:,t,:] = x.squeeze() #B*512
        out, _ = self.rnn(input, None)
        x = self.fc(out[:, -1, :])
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class WindModelE2E(nn.Module):
    def __init__(self,arch, num_classes, lstm_layers, hidden_size):
        super(WindModelE2E, self).__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.epsilon = 1e-4
        # self.OpticalNet = PWCNetwork()
        self.Classifier = CNN_LSTM(self.arch, self.num_classes, self.lstm_layers, self.hidden_size).cuda()
        # self.Classifier_rgb = CNN_LSTM_rgb(self.arch, self.num_classes, self.lstm_layers, self.hidden_size).cuda()
        # self.fuse_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # self.fuse_weight_relu = nn.ReLU()
        # self.dropout = nn.Dropout()
        self.ARFlow = PWCLite().cuda()
        # self.swish = Swish()
        # self.fc = nn.Linear(self.num_classes, self.num_classes)

    def arflow(self, tensorFirst, tensorSecond): #b, 3, h ,w
        assert (tensorFirst.size(2) == tensorSecond.size(2))
        assert (tensorFirst.size(3) == tensorSecond.size(3))

        intWidth = tensorFirst.size(2)
        intHeight = tensorFirst.size(3)
        batch_size = len(tensorFirst)

        # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
        # assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

        tensorPreprocessedFirst = tensorFirst.cuda()  # .view(1, 3, intHeight, intWidth)
        tensorPreprocessedSecond = tensorSecond.cuda()  # .view(1, 3, intHeight, intWidth)

        img_pair = torch.cat([tensorPreprocessedFirst, tensorPreprocessedSecond], 1)

        res_dict = self.ARFlow(img_pair, with_bk=True)

        flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
        flows_1221 = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                    zip(flows_12, flows_21)] # 多层输出的flows， 5 * b, 2+2, 256/128/.., 256/128/..


        flow_to_image = res_dict['flows_fw'][0] # b, 2, 256, 256

        flows = torch.zeros(batch_size, 3, intWidth, intHeight)
        for i in range(batch_size):
            flow = show_flow_hsv(flow_to_image[i, :, :, :].permute(1, 2, 0).cpu().detach().numpy())
            flows[i, ...] = torch.FloatTensor(flow).permute(2,0,1)

        return flows, flows_1221

    # def estimateOpticalFlow(self, tensorFirst, tensorSecond):  # 10,3,w,h
    #     assert (tensorFirst.size(2) == tensorSecond.size(2))
    #     assert (tensorFirst.size(3) == tensorSecond.size(3))

    #     intWidth = tensorFirst.size(2)
    #     intHeight = tensorFirst.size(3)
    #     batch_size = len(tensorFirst)

    #     # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    #     # assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    #     tensorPreprocessedFirst = tensorFirst  # .view(1, 3, intHeight, intWidth)
    #     tensorPreprocessedSecond = tensorSecond  # .view(1, 3, intHeight, intWidth)

    #     intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    #     intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    #     tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst,
    #                                                               size=(intPreprocessedHeight, intPreprocessedWidth),
    #                                                               mode='bilinear', align_corners=False)
    #     tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond,
    #                                                                size=(intPreprocessedHeight, intPreprocessedWidth),
    #                                                                mode='bilinear', align_corners=False)

    #     optical_flow = self.OpticalNet(tensorPreprocessedFirst, tensorPreprocessedSecond)
    #     tensorFlow = 20.0 * torch.nn.functional.interpolate(input=optical_flow, size=(intHeight, intWidth),
    #                                                         mode='bilinear', align_corners=False)

    #     tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    #     tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    #     # tensorOutput = tensorFlow[0, :, :, :]
    #     # flow = tensorOutput.permute(1, 2, 0)
    #     # flow2 = self.show_flow_hsv(flow.cpu().detach().numpy())

    #     #flow = [show_flow_hsv(tensorFlow[i, :, :, :].permute(1, 2, 0).cpu().detach().numpy()).transpose(2, 1, 0)
    #     #        for i in range(batch_size)]
    #     flows = torch.zeros(batch_size, 3, tensorFlow.size(2),tensorFlow.size(3) )
    #     for i in range(batch_size):
    #         flow = show_flow_hsv(tensorFlow[i, :, :, :].permute(1, 2, 0).cpu().detach().numpy())
    #         flows[i, ...] = torch.FloatTensor(flow).permute(2,0,1)

    #     return flows

    def forward(self, inputs): #inputs will be a batch of 10 frames of a video, 10*3*H*W
        batch_size = len(inputs) #number of batch
        length = inputs.size(1) #number of frames
        w = inputs.size(3)
        h = inputs.size(4)
        ofs = torch.zeros(batch_size, length-1, 3,w,h).cuda()
        flows_1221_frame =[]
        for i in range(length-1):
            first = inputs[:, i].clone().detach()
            second = inputs[:, i+1].clone().detach()
            # flow = self.estimateOpticalFlow(first, second)
            flow, flows_1221 = self.arflow(first, second)
            ofs[:,i,:,:,:] = torch.FloatTensor(flow)
            flows_1221_frame.append(flows_1221)
        
        outputs = self.Classifier(ofs)


        return outputs, flows_1221_frame



if __name__ == '__main__':
    # model = CNN_LSTM('resnet18', 11, 1, 128)

    model = WindModelE2E('resnet18', 11, 1, 128)
    model.cuda()

    inputs = torch.randn(8, 10, 3, 256, 256).cuda()
    with autocast():
        output, flows_1221_frame = model(inputs)

    summary(model, torch.zeros(1, 10, 3, 256, 256).cuda())
    print(output.shape)
