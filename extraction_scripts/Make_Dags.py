#Make dagman sibmit files for corsika
Set = 20789
Folder = 100
Num = 100
#name = "/scratch/dpankova/corsika_dags/Make_Images_DAGS_Set{0}_Folders{1:d}_Number{2:d}.dag".format(Set,Folder,Num)
name = "/home/dpankova/work/double_pulse/Make_Images_DAGS_Set{0}_Folders{1:d}_Number{2:d}.dag".format(Set,Folder,Num)
print(name)
myfile = open(name, 'w')
for f in range(0,Folder):
    for n in range(0,Num):
        job_name = "s{0}f{1:d}n{2:d}".format(Set,f,n)
        job = "JOB {0:s} /home/dpankova/work/double_pulse/Make_Images_Corsika.sub".format(job_name)
#        job = "JOB {0:s} /data/user/dpankova/double_pulse/Make_Images_MuonGun.sub".format(job_name)
#        job = "JOB {0:s} /data/user/dpankova/double_pulse/Make_Images_Corsika_100.sub".format(job_name)
        var1 = "VARS {0:s} dset = \"{1:d}\"".format(job_name,Set)
        var2 = "VARS {0:s} folder = \"{1:d}\"".format(job_name,f)
        var3 = "VARS {0:s} number = \"{1:d}\"".format(job_name,n)
        print(job)
        print(var1)
        print(var2)
        print(var3)
        myfile.write("%s\n" % job)
        myfile.write("%s\n" % var1)
        myfile.write("%s\n" % var2)
        myfile.write("%s\n" % var3)
myfile.close()
