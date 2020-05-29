#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT /data/user/dpankova/oscNext/oscNext_meta/build/ 

python /data/user/dpankova/double_pulse/Make_Images_new.py  -it 1 -o Genie_Make_image_TEST -i /data/ana/Cscd/StartingEvents/NuGen_new/NuTau/medium_energy/IC86_flasher_p1=0.3_p2=0.0/l2/1/l2_00000001.i3.zst -gcd /data/user/dpankova/double_pulse/GeoCalibDetectorStatus_2013.56429_V1_Modified.i3.gz -t genie 
