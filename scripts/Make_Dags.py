#Make dagman sibmit files for corsika
Set = 11058
Folder = 32
Num = 1000
name = "/scratch/dpankova/corsika_dags/Make_Images_DAGS_Set{0}_Folders{1:d}_Number{2:d}.dag".format(Set,Folder,Num)
print(name)
myfile = open(name, 'w')
for f in range(0,Folder):
    for n in range(0,Num):
        job_name = "{0}_{1:d}_{2:d}".format(Set,f,n)
        job = "JOB {0:d} /data/user/dpankova/double_pulse/Make_Images_Corsika.sub".format(f*Num+n)
        var1 = "VARS {0:d} dset = \"{1:d}\"".format(f*Num+n,Set)
        var2 = "VARS {0:d} folder = \"{1:d}\"".format(f*Num+n,f)
        var3 = "VARS {0:d} number = \"{1:d}\"".format(f*Num+n,n)
        print(job)
        print(var1)
        print(var2)
        print(var3)
        myfile.write("%s\n" % job)
        myfile.write("%s\n" % var1)
        myfile.write("%s\n" % var2)
        myfile.write("%s\n" % var3)
myfile.close()
