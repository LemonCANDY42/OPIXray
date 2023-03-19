# -*- coding: utf-8 -*-
# @Time    : 2022/8/30 18:42
# @Author  : Kenny Zhou
# @FileName: makeTxt.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com
import os
import random
from pathlib import Path
from OPIXray.utils.file_tools import flexible_open

trainval_percent = 1
train_percent = 0.8
xmlfilepath = '/home/data/1284'
txtsavepath = 'dataset/ImageSets'
total_files = Path(xmlfilepath)
total_xml = list(total_files.glob('**/*.xml'))
num = len(total_xml)
num_list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(num_list, tv)
train = random.sample(trainval, tr)
ftrainval = flexible_open('dataset/ImageSets/trainval.txt', 'w')
ftest = flexible_open('dataset/ImageSets/test.txt', 'w')
ftrain = flexible_open('dataset/ImageSets/train.txt', 'w')
fval = flexible_open('dataset/ImageSets/val.txt', 'w')

print(len(trainval),len(train))
for i in num_list:
    name = total_xml[i].stem
    # print(name)
    if i in trainval:
        ftrainval.write(name)
        ftrainval.write('*')
        if i in train:
            ftrain.write(name)
            ftrain.write('*')
        else:
            fval.write(name)
            fval.write('*')
    else:
        ftest.write(name)
        ftest.write('*')

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
