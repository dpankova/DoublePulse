#!/usr/bin/env 

import numpy as np
import glob
import re

bs_runs = []

f = open('BurnSampleRuns.txt', 'r')
lines = f.readlines()
for line in lines[:]:
    sp = line.split()
    bs_runs.append(int(sp[0])) 

print(len(bs_runs),bs_runs)

gr_files = glob.glob('./IC86_*_GoodRunInfo.txt')

runs = []
for filename in gr_files:
    f = open(filename, 'r')
    lines = f.readlines()
    for line in lines[2:]:
        sp = line.split()
        if not (int(sp[0]) in bs_runs) and int(sp[1])==1: 
            runs.append(line)
    f.close()

myfile = open("DataRuns.txt", 'w')
for i in runs:
    myfile.write(i)
myfile.close()

print("finished",len(runs))
