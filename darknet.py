import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import os
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt


def parse_cfg(config_file):
    file = open(config_file, 'r')
    lines = file.read().split('\n')
    lines = [line for line in lines if len(line) > 0 and line[0] != '#']
    lines = [line.lstrip().rstrip() for line in lines]

    output = []
    element_dict = {}
    for line in lines:

        if line[0] == '[':
            if len(element_dict) != 0:  # appending the dict stored on previous iteration
                output.append(element_dict)
                element_dict = {}  # again emtying dict
            element_dict['type'] = ''.join([i for i in line if i != '[' and i != ']'])

        else:
            val = line.split('=')
            element_dict[val[0].rstrip()] = val[1].lstrip()  # removing spaces on left and right side

    output.append(element_dict)  # appending the values stored for last set

    return output


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_model(blocks):
    darknet_details = blocks[0]
    channels = 3
    output_filters = []
    module_list = nn.ModuleList()

    for i, block in enumerate(blocks[1:]):
        seq = nn.Sequential()
        if block["type"] == "convolutional":
            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])
            use_bias = False if ("batch_normalize" in block) else True
            pad = (kernel_size - 1) // 2

            conv = nn.Conv2d(in_channels=channels, out_channels=filters, kernel_size=kernel_size,
                             stride=strides, padding=pad, bias=use_bias)
            seq.add_module("conv_{0}".format(i), conv)

            if "batch_normalize" in block:
                bn = nn.BatchNorm2d(filters)
                seq.add_module("batch_norm_{0}".format(i), bn)

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                seq.add_module("leaky_{0}".format(i), activn)

        elif block["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            seq.add_module("upsample_{}".format(i), upsample)

        elif block["type"] == 'route':
            # start and end is given in format (eg:-1 36 so we will find layer number from it.
            # we will find layer number in negative format
            # so that we can get the number of filters in that layer
            block['layers'] = block['layers'].split(',')
            block['layers'][0] = int(block['layers'][0])
            start = block['layers'][0]
            if len(block['layers']) == 1:
                filters = output_filters[i + start]


            elif len(block['layers']) > 1:
                block['layers'][1] = int(block['layers'][1]) - i
                end = block['layers'][1]
                filters = output_filters[i + start] + output_filters[i + end]

            route = EmptyLayer()
            seq.add_module("route_{0}".format(i), route)


        elif block["type"] == "shortcut":
            from_ = int(block["from"])
            shortcut = EmptyLayer()
            seq.add_module("shortcut_{0}".format(i), shortcut)


        elif block["type"] == "yolo":
            mask = block["mask"].split(",")
            mask = [int(m) for m in mask]
            anchors = block["anchors"].split(",")
            anchors = [(int(anchors[i]), int(anchors[i + 1])) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            block["anchors"] = anchors

            detectorLayer = DetectionLayer(anchors)
            seq.add_module("Detection_{0}".format(i), detectorLayer)

        module_list.append(seq)
        output_filters.append(filters)
        channels = filters

    return darknet_details, module_list


def prediction(x, inp_dim, anchors, num_classes, CUDA=True):
    # the idea of this function is from
    # https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/

    # x --> 4D feature map
    batch_size = x.size(0)
    grid_size = x.size(2)
    stride = inp_dim // x.size(2)  # factor by which current feature map reduced from input
    #     grid_size = inp_dim // stride

    bbox_attributes = 5 + num_classes
    num_anchors = len(anchors)
    #
    prediction = x.view(batch_size, bbox_attributes * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attributes)

    # the dimension of anchors is wrt original image.We will make it corresponding to feature map
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)  # (1,gridsize*gridsize,1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors  # width and height
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))
    prediction[:, :, :4] *= stride
    return prediction


class Darknet(nn.Module):
    # the idea of this function is from
    # https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_model(self.blocks)

    def forward(self, x, CUDA=False):
        modules = self.blocks[1:]
        outputs = {}  # We cache the outputs for the route layer
        write = 0  # This is explained a bit later
        for i, module in enumerate(modules):
            module_type = (module["type"])
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
                outputs[i] = x

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                if len(layers) > 1:
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

                outputs[i] = x

            elif module_type == "shortcut":
                from_ = int(module["from"])

                # just adding outputs for residual network
                x = outputs[i - 1] + outputs[i + from_]
                outputs[i] = x

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors

                # Get the input dimensions
                inp_dim = int(self.net_info["height"])
                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform
                x = x.data  # get the data at that point
                x = prediction(x, inp_dim, anchors, num_classes)

                if not write:  # if no collector has been intialised.
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i - 1]

        try:
            return detections  # return detections if present
        except:
            return 0

    def load_weights(self, weightfile):

        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 4 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4. IMages seen
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
                # Note: we dont have bias for conv when batch normalization is there