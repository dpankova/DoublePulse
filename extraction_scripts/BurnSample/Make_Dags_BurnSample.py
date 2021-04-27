#Make dagman sibmit files for corsika
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
            runs.append([sp[0],sp[7]])
    f.close()

l = len(runs)
print(l)
name = "/scratch/dpankova/corsika_dags/Make_Images_BurnSample_test.dag"
#name = "./Make_Images_BurnSample.dag"
print(name)
bad_runs = 0
size = 0
#myfile = open(name, 'w')
for r in range(l):
    run_files = glob.glob(runs[r][1]+"*Subrun*.i3.*")
    gcd_files = glob.glob(runs[r][1]+"*GCD*.i3.*")
    size+=len(run_files)
    if len(run_files) == 0 or len(gcd_files) == 0:
        bad_runs += 1
        continue
    for n,f in enumerate(run_files[:1]):
        job_name = "r{0:s}f{1:d}".format(runs[r][0],n)
        job = "JOB {0:s} /home/dpankova/work/double_pulse/Make_Images_BurnSample.sub".format(job_name)
        var1 = "VARS {0:s} infile = \"{1:s}\"".format(job_name,f)
        var2 = "VARS {0:s} gcdfile = \"{1:s}\"".format(job_name,gcd_files[0])
        var3 = "VARS {0:s} run = \"{1:s}\"".format(job_name,runs[r][0])
        var4 = "VARS {0:s} file = \"{1:d}\"".format(job_name,n)
#        print(job)
#        print(var1)
#        print(var2)
#        print(var3)
#        print(var4)
#        myfile.write("%s\n" % job)
#        myfile.write("%s\n" % var1)
#        myfile.write("%s\n" % var2)
#        myfile.write("%s\n" % var3)
#        myfile.write("%s\n" % var4)
#myfile.close()
print(size,bad_runs)
