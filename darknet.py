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