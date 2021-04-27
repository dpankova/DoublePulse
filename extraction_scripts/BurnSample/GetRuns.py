#!/usr/bin/env 

import numpy as np
import glob
import re

gr_files = glob.glob('./IC86_*_GoodRunInfo.txt')

runs = []
for filename in gr_files:
    f = open(filename, 'r')
    lines = f.readlines()
    for line in lines[2:]:
        sp = line.split()
        if (int(sp[0]) % 100 ==0) and int(sp[1])==1 and int(sp[2])==1: 
            runs.append(line)
    f.close()

myfile = open("BurnSampleRuns.txt", 'w')
for i in runs:
    myfile.write(i)
myfile.close()

print("finished")
