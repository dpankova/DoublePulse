#Make summarize the txt file from image extraction
import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d","--directory",
                    dest="directory",
                    type=str,
                    nargs = "+",
                    help="Directory with txt files")

parser.add_argument("-i3","--i3directory",
                    dest="i3directory",
                    type=str,
                    nargs = "+",
                    help="Directory with i3 files")

parser.add_argument("-o","--outfile",
                    dest="outfile",
                    type=str,
                    default="Out",
                    help="base name for outfile")

args = parser.parse_args()
directory=args.directory
i3directory=args.i3directory
outfile=args.outfile

print(i3directory)
#i3files = set(glob.glob(i3directory[0]+'/?????-?????/*.i3.bz2'))
i3files = set(glob.glob(i3directory[0]+'/???????-???????/*.i3.zst'))
i3_files = len(i3files)
print(i3_files)
#print(total_txt_files)


directory = directory[0]+"*_text.txt"
txt_files = len(glob.glob(directory))
files_started = 0
files_failed = 0
empty_files = 0
event_count = 0
for filename in glob.glob(directory):
    f = open(filename, 'r')
    lines = f.readlines()
    if len(lines) == 0:
        print("empty file =", filename)
        empty_files += 1
        continue
    if len(lines) == 1:
        print("Job didn't finish. {0:d} lines in file {1}".format(len(lines),filename))
        name = lines[0].rstrip()
        if not name in i3files:
            print("Error: this file is not among i3 files {0:s}".format(name))
            break
        files_started +=1
        files_failed +=1
        continue
    if len(lines) > 2:
        print("{0:d} lines in file {1}".format(len(lines),filename))
        lines = lines[-2:]

    name = lines[0].rstrip()
    count = lines[1].rstrip()
    if not count.isdigit():
        print("Job didn't finish. {0:d} lines in file {1}".format(len(lines),filename))
        if not count in i3files:
            print("Error: this file is not among i3 files {0:s}".format(name))
            break
        files_started +=1
        files_failed +=1
        continue
        
    if not name in i3files:
        print("Error: this file is not among i3 files {0:s}".format(name))
        break
    files_started += 1
    event_count +=int(count)
    f.close()

myfile = open(outfile, 'w')
myfile.write("Total i3 files = %d\n" % i3_files)
myfile.write("Total txt files = %d\n" % txt_files)
myfile.write("Files started %d\n" % files_started)
myfile.write("Files failed %d\n" % files_failed)
myfile.write("Empty files %d\n" % empty_files)
myfile.write("Event count %d\n" % event_count)
myfile.write("For weights %d\n" % (files_started-files_failed))
myfile.close()
print("finished")
