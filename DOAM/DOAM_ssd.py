# -*- coding: utf-8 -*-
# @Time    : 2022/9/4 21:34
# @Author  : Kenny Zhou
# @FileName: DOAM_ssd.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com
import torch
import torch.nn as nn
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

for name in ssd_model.state_dict():
	print(name)

torch.save(ssd_model,"./nvidia_ssd")
torch.save(utils,"./nvidia_ssd_utils")


class DOAM_SSD(nn.Module):

	def __init__(self, phase, size, base, extras, head, num_classes, mode='cuda', type='ssd', ft_module=None,
							 pyramid_ext=None, ssd_model=ssd_model):
		super(SSD, self).__init__()
		self.phase = phase
		self.num_classes = num_classes
		self.cfg = (coco, DongYing)[num_classes == 21]
		self.priorbox = PriorBox(self.cfg)
		self.priors = Variable(self.priorbox.forward(), volatile=True)
		self.size = size
		self.model_type = type

		# SSD network
		self.ssd_model = ssd_model

		self.edge_conv2d = DOAM(mode=mode)

		if phase == 'test':
			self.softmax = nn.Softmax(dim=-1)
			self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
		if phase == 'onnx':
			self.softmax = nn.Softmax(dim=-1)
		# self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

	def forward(self, x):

		# apply vgg up to conv4_3 relu
		x = self.edge_conv2d(x)

		output = self.ssd_model(x)
		# print(output)
		return output


def build_ssd(phase, size=300, num_classes=21, mode=None, type="ssd"):
	return DOAM_SSD(phase, size, base_, extras_, head_, num_classes, mode, type=type, ft_module=layers,
									pyramid_ext=pyramid_ext)
