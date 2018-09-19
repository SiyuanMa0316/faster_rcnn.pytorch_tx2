# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False, channel_num=64):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = channel_num
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    #vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    #self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
    vgg.classifier = nn.Sequential(nn.Linear(in_features=self.dout_base_model*49, out_features=self.dout_base_model*8, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.5),
                                       nn.Linear(in_features=self.dout_base_model*8, out_features=self.dout_base_model*8, bias=True),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(p=0.5))

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.ReLU(inplace=True),
                                       #nn.Conv2d(64, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       #nn.ReLU(inplace=True),
                                       #nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1),
                                       #         ceil_mode=False),
                                       nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1),
                                                ceil_mode=False),
                                       nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1),
                                                    ceil_mode=False),
                                       nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1),
                                                    ceil_mode=False),
                                       nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1),
                                                    ceil_mode=False),
                                       nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(512, self.dout_base_model, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.ReLU(inplace=True))


    # Fix the layers before conv3 before=10:
    for layer in range(5):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(self.dout_base_model*8, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(self.dout_base_model*8, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(self.dout_base_model*8, 4 * self.n_classes)

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

