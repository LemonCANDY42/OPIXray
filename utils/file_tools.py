# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 12:32
# @Author  : Kenny Zhou
# @FileName: file_tools.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com
import os
from pathlib import Path


def flexible_open(path,*args,**kwargs):
	p = Path(path)
	if not p.exists():
		p.mkdir(parents=True, exist_ok=True)
	return open(p.absolute(),*args,**kwargs)
