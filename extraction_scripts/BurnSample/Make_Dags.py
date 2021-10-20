#Make dagman sibmit files for corsika
import numpy as np
import glob
import re

bs_runs = []
f = open('BurnSampleRuns.txt', 'r')
lines = f.readlines()
for line in lines:
    sp = line.split()
    bs_runs.append(int(sp[0]))


gr_files = glob.glob('./IC86_2020_GoodRunInfo.txt')

runs = [] #selected runs
time = 0 #total livetime
for filename in gr_files:
    f = open(filename, 'r')
    lines = f.readlines()
    for line in lines:
        if not line.startswith('/data/exp/IceCube/'): # pass header lines
            continue
        sp = line.split()
        if int(sp[1])==1 and not (int(sp[0]) in bs_runs): #take runs that end with 00, good_i3
            if sp[7][-1] == '/':
                runs.append([sp[0],sp[7]]) #save run number and path to files
            else:
                runs.append([sp[0],sp[7]+'/']) #save run number and path to files
            #if not sp[0] in empty_runs: #if not empty
            time += float(sp[3]) #save livetime
    f.close()

print('live time = ',time)
print('number of runs = ', len(runs))

name = "/scratch/dpankova/corsika_dags/Make_Images_Data2020_100.dag"
#name = "./Make_Images_BurnSample.dag"

bad_runs_list = []
i3files = 0

#myfile = open(name, 'w')
for r in runs:
    run_files = []
    run_files_all = glob.glob(r[1]+"*Subrun*.i3.*") #i3files
    for run in run_files_all:
        if not "_IT.i3." in run:
            run_files.append(run)

    gcd_files = glob.glob(r[1]+"*GCD*.i3.*") #GCD
    i3files+=len(run_files)

    if len(run_files) == 0 or len(gcd_files) == 0:
        bad_runs_list.append([r[0]])
        continue

    dec = len(run_files)//100 #one job for 10 i3 files
    for i in range(dec+1):
        job_name = "r{0:s}f{1:d}".format(r[0],i)
        job = "JOB {0:s} /data/user/dpankova/double_pulse/BurnSample/Make_Images_Data.sub".format(job_name)
        var1 = "VARS {0:s} infile = \"{1:s}{2:02d}*.i3.zst\"".format(job_name,run_files[0][:-11],i) 
        var2 = "VARS {0:s} gcdfile = \"{1:s}\"".format(job_name,gcd_files[0])
        var3 = "VARS {0:s} run = \"{1:s}\"".format(job_name,r[0])
        var4 = "VARS {0:s} file = \"{1:d}\"".format(job_name,i*100)
#        ss = "{0:s}{1:02d}*.i3.zst".format(run_files[0][:-10],i)
#        ff = glob.glob(ss)
#        print(len(ff))
#         print(job)
#         print(var1)
#         print(var2)
#         print(var3)
#         print(var4)
#         myfile.write("%s\n" % job)
#         myfile.write("%s\n" % var1)
#         myfile.write("%s\n" % var2)
#         myfile.write("%s\n" % var3)
#         myfile.write("%s\n" % var4)
# myfile.close()

print('number of i3 files =',i3files)
print('bad runs =', len(bad_runs_list), bad_runs_list)
