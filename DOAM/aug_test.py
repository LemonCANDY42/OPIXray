# -*- coding: utf-8 -*-
# @Time    : 2022/9/4 18:53
# @Author  : Kenny Zhou
# @FileName: aug_test.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com
BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White
import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
	"""Visualizes a single bounding box on the image"""
	# x_min, y_min, w, h = bbox
	x_min, y_min, x_max, y_max = bbox#int(x_min), int(x_min + w), int(y_min), int(y_min + h)
	x_min, y_min, x_max, y_max = int(x_min*504),int(y_min*576),int(x_max*504),int(y_max*576)
	cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

	((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
	cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
	cv2.putText(
		img,
		text=class_name,
		org=(x_min, y_min - int(0.3 * text_height)),
		fontFace=cv2.FONT_HERSHEY_SIMPLEX,
		fontScale=0.35,
		color=TEXT_COLOR,
		lineType=cv2.LINE_AA,
	)
	return img

def visualize(image, bboxes, category_ids, category_id_to_name):
	img = image.copy()
	for bbox, category_id in zip(bboxes, category_ids):
			class_name = category_id_to_name[category_id]
			img = visualize_bbox(img, bbox, class_name)
	plt.figure(figsize=(12, 12))
	plt.axis('off')
	plt.imshow(img)
	plt.show()

if __name__ == "__main__":

	image = cv2.imread('/Users/kennymccormick/github/Seeed_SMG_AIOT/OPIXray/images/DYAJJ_20220727_V19_p_train_conveyor_belt_13_16560.jpg')
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	bboxes = [[255/504, 280/576, 384/504, 406/576]]
	category_ids = [0]

	# We will use the mapping from category_id to the class name
	# to visualize the class label for the bounding box on the image
	category_id_to_name = {0: 'qiang'}
	visualize(image, bboxes, category_ids, category_id_to_name)

	transform = A.Compose([
		A.HorizontalFlip(p=0.5),
		A.VerticalFlip(p=0.5),
		A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3),  # 随机明亮对比度
	], bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids']))
	# random.seed(7)
	transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
	visualize(
			transformed['image'],
			transformed['bboxes'],
			transformed['category_ids'],
			category_id_to_name,
	)
