"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from enum import Enum, unique
import os
# from .config import HOME
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
from PIL import Image
from pathlib import Path
import numpy as np
from OPIXray.DOAM.utils.augmentations import resize_image,augmt_transformed
from torchvision import transforms as T

if sys.version_info[0] == 2:
	import xml.etree.cElementTree as ET
else:
	import xml.etree.ElementTree as ET

# OPIXray_CLASSES = (
# 	'Folding_Knife', 'Straight_Knife', 'Scissor', 'Utility_Knife', 'Multi-tool_Knife',
# )
OPIXray_CLASSES = (
		'banshou', 'bianpao', 'bijibendiannao', 'boliping', 'caidao',
		'chuizi', 'dahuoji', 'dangong', 'danzhu', 'dianchi', 'futou',
		'gongyidao', 'gunbang', 'jiandao', 'juzi', 'logo', 'paozhang',
		'penwu', 'qiang', 'qianzi', 'quanci', 'rongqi', 'shouji', 'shoukao',
		'xiaodao', 'yanhua', 'zhediedao', 'zhihu', 'zhuizi'
)
OPIXray_ROOT = "OPIXray_Dataset/train/"


@unique
class LabelType(Enum):
	OPIXray = 0
	DongYing = 1

class OPIXrayAnnotationTransform(object):
	"""Transforms a VOC annotation into a Tensor of bbox coords and label index
	Initilized with a dictionary lookup of classnames to indexes

	Arguments:
			class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
					(default: alphabetic indexing of VOC's 20 classes)
			keep_difficult (bool, optional): keep difficult instances or not
					(default: False)
			height (int): height
			width (int): width
	"""

	def __init__(self, class_to_ind=None, keep_difficult=False):
		self.class_to_ind = class_to_ind or dict(
			zip(OPIXray_CLASSES, range(len(OPIXray_CLASSES))))
		self.keep_difficult = keep_difficult
		self.type_dict = {}
		self.type_sum_dict = {}
		self.dataset_type = "OPIXary"

	def set_dataset_type(self,dataset_type:str):
		self.dataset_type=dataset_type

	def __call__(self, target, width, height, idx):
		"""
		Arguments:
				target (annotation) : the target annotation to be made usable
						will be an ET.Element
				it has been changed to the path of annotation-2019-07-10
		Returns:
				a list containing lists of bounding boxes  [bbox coords, class name]
		"""
		# print (idx)
		res = []
		with open(target, "r", encoding='utf-8') as f1:
			dataread = f1.readlines()
		for annotation in dataread:
			bndbox = []
			temp = annotation.split()

			if self.dataset_type == "OPIXary":

				name = temp[1]
				if name not in OPIXray_CLASSES:
					continue
				xmin = int(temp[2]) / width
				if xmin > 1:
					continue
				if xmin < 0:
					xmin = 0
				ymin = int(temp[3]) / height
				if ymin < 0:
					ymin = 0
				xmax = int(temp[4]) / width
				if xmax > 1:
					xmax = 1
				ymax = int(temp[5]) / height
				if ymax > 1:
					ymax = 1
				bndbox.append(xmin)
				bndbox.append(ymin)
				bndbox.append(xmax)
				bndbox.append(ymax)

			else:

				name = temp[0]
				if name not in OPIXray_CLASSES:
					continue
				xmin = float(temp[1]) #/ width
				# if xmin > 1:
				# 	continue
				if xmin < 0:
					xmin = 0
				ymin = float(temp[2]) #/ height
				if ymin < 0:
					ymin = 0
				xmax = float(temp[3]) #/ width
				# if xmax > 1:
				# 	xmax = 1
				ymax = float(temp[4]) #/ height
				# if ymax > 1:
				# 	ymax = 1
				bndbox.append(xmin)
				bndbox.append(ymin)
				bndbox.append(xmax)
				bndbox.append(ymax)

			label_idx = self.class_to_ind[name]
			# label_idx = name
			bndbox.append(label_idx)
			res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
		if len(res) == 0:
			return [[0, 0, 0, 0, len(OPIXray_CLASSES)]]
		return res


def test_Sobel(img):
	# src = cv2.imread(src)
	# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
	y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
	absX = cv2.convertScaleAbs(x)
	absY = cv2.convertScaleAbs(y)
	dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
	return dst


class OPIXrayDetection(data.Dataset):
	def __init__(self,
							 image_sets=None,
							 root=OPIXray_ROOT,
							 transform=None, target_transform=OPIXrayAnnotationTransform(), phase=None, type=LabelType.OPIXray):
		'''

		Args:
				image_sets: image_sets need set value!
				root:
				transform:
				target_transform:
				phase:
		'''
		self.root = root
		self.image_set = image_sets
		self.transform = transform
		self.target_transform = target_transform
		self.type = type
		self.target_transform.set_dataset_type(self.type.name)
		# self.name = dataset_name
		self.name = self.type.name
		self.as_tensor = T.Compose([
			T.ToTensor(),
		])
		self.phase = phase

		if (phase == 'test'):
			if self.type is LabelType.OPIXray:
				self._annopath = os.path.join('%s' % self.root, '{}_annotation'.format(phase), '%s.txt')
				# self._imgpath = os.path.join('%s' % self.root, 'test_image', '%s.TIFF')
				# self._imgpath1 = os.path.join('%s' % self.root, 'test_image', '%s.tiff')
				self._imgpath2 = os.path.join('%s' % self.root, '{}_image'.format(phase), '%s.jpg')
			else:
				self.images_file_path = Path(f'{self.root}') / Path(f'./{phase}.txt')
				self.labels_folder = Path(f'{self.root}') / Path('./labels')

		elif (phase == 'train'):
			if self.type is LabelType.OPIXray:
				self._annopath = os.path.join('%s' % self.root, '{}_annotation'.format(phase), '%s.txt')
				# self._imgpath = os.path.join('%s' % self.root, 'train_image', '%s.TIFF')
				# self._imgpath1 = os.path.join('%s' % self.root, 'train_image', '%s.tiff')
				self._imgpath2 = os.path.join('%s' % self.root, '{}_image'.format(phase), '%s.jpg')
			else:
				self.images_file_path = Path(f'{self.root}') / Path(f'./{phase}.txt')
				self.labels_folder = Path(f'{self.root}') / Path('./labels')

		else:
			print('No phase')
		self.ids = list()

		# listdir = os.listdir(os.path.join('%s' % self.root, 'Annotation'))
		if self.type is LabelType.OPIXray:
			if self.image_set:
				with open(self.image_set, 'r') as f:
					lines = f.readlines()
					for line in lines:
						self.ids.append(line.strip('\n'))
		else:
			if (phase == 'train'):
				if self.images_file_path:
					with open(self.images_file_path, 'r') as f:
						lines = f.readlines()
						for line in lines:
							self.ids.append(Path(line.strip('\n')))

	def __getitem__(self, index):
		im, gt, h, w, og_im = self.pull_item(index)

		return im, gt

	def __len__(self):
		return len(self.ids)

	def pull_item(self, index):
		img_id = self.ids[index]

		# target = ET.parse(self._annopath % img_id).getroot()
		if self.type is LabelType.OPIXray:
			target = self._annopath % img_id
			img_path = self._imgpath2 % img_id
			img = cv2.imread(img_path)
		else:
			img_path = img_id
			if (self.phase == 'train'):
				target = self.labels_folder / img_id.stem
				target = str(target)+'.txt'
			# img = cv2.imread(str(img_path))
			img = Image.open(str(img_path))
			img = np.asarray(img)
			if (self.phase == 'test'):
				img = cv2.resize(img, (300, 300))
				print(img.shape)

		if img is None:
			raise 'wrong'

		try:
			height, width, channels = img.shape
		except:
			# print(img_id)
			raise f'can\'t get {img_path} shape'
		# print("height: " + str(height) + " ; width : " + str(width) + " ; channels " + str(channels) )
		og_img = img
		# yuv_img = cv2.cvtColor(og_img,cv2.COLOR_BGR2YUV)

		# try:
		# 	img = cv2.resize(img, (300, 300))
		# except:
		# 	print('img_read_error')

		# img = np.concatenate((img,sobel_img),2)
		# print (img_id)


		if (self.phase == 'train'):
			if self.target_transform is not None:
				target = self.target_transform(target, width, height, img_id)

			# if target[0] != [0, 0, 0, 0, len(OPIXray_CLASSES)]:
			augments = augmt_transformed(img,target)
			img = augments['image']
			target = augments['bboxes']
			# else:
			# 	print(target)
			transformed_dict = resize_image(img,target, 300, 300)
			target = transformed_dict['bboxes']
			# transformed_dict = resize_image(img,target, 300, 300)

			# # contains the image as array
			img = self.as_tensor(transformed_dict["image"])
			# # contains the resized bounding boxes
			# print(target)
			if target != []:
				temp_t = []
				for t in target:
					temp_t.append([t[0]/300,t[1]/300,t[2]/300,t[3]/300,t[-1]])
				target = np.array(list(map(list, temp_t))).astype(float)
			else:
				target = np.array([[0., 0., 0., 0., len(OPIXray_CLASSES)]]).astype(float)
		else:
			img = torch.from_numpy(img)
			target = None
		# print(target,type(target))

		#'''
		# if self.transform is not None:
		# 		target = np.array(target)
		# 		#print('\n\n\n' + str(target) + '\n\n\n' )
		# 		img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
		# 		# to rgb
		# 		img = img[:, :, (2, 1, 0)]
		# 		# img = img.transpose(a2, 0, a1)
		# 		target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
		#'''

		# return torch.from_numpy(img).permute(2, 0, 1), target, height, width, og_img
		return img, target, height, width, og_img
