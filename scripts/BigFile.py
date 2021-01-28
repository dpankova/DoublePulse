#!/usr/bin/python
from __future__ import division
import numpy as np
import glob
import sys
import pickle

size = 0
types = [["NuECC", 0],["NuENC", 1],["NuMuCC", 2],["NuMuNC", 3],["NuTauCC", 4],["NuTauNC", 5]]
for t in types:
    for name in glob.glob('/home/dup193/work/double_pulse/data/Nu_all/'+t[0]+'*_data.npy'):
        x = np.load(name, mmap_mode="r")
        print(name, x.shape, t[1])
        h,w,d = x.shape
        size = size +h
print(size)

pos = 0
nu_data = np.memmap('/fastio2/dasha/double_pulse/nu_data.npy', mode = 'w+', dtype ='float32', shape=(size,300,60))
nu_label = np.memmap('/fastio2/dasha/double_pulse/nu_label.npy', mode = 'w+', dtype ='float32', shape=size)
nu_files = []

for t in types:
    for name in glob.glob('/home/dup193/work/double_pulse/data/Nu_all/'+t[0]+'*_data.npy'):
        data = np.load(name, mmap_mode="r")
        file_name = name[43:-9]
        h,w,d = data.shape
        label = t[1]
        nu_files.append([file_name,pos])
        print(file_name, data.shape, label, pos)
        
        data = np.float32(data)
        nu_data[pos:pos+h] = data
        nu_label[pos:pos+h] = np.full(shape=h,fill_value=label,dtype=np.int)
        pos = pos + h
        
np.save('/fastio2/dasha/double_pulse/nu_files.npy',nu_files)
print(len(nu_files))