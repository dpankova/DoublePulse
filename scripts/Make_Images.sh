#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT /data/user/dpankova/oscNext/oscNext_meta/build/ 

#python /data/user/dpankova/double_pulse/Make_Images.py  -o Corsika_Make_image_TEST -i /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11057/00000-00999/Level2_IC86.2012_corsika.011057.000000.i3.bz2 -gcd /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11057/01000-01999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz -t corsika -set 11057 

#python /data/user/dpankova/double_pulse/Make_Images.py -it 1 -o Diff_Test -i ./old/Diff_NuECC_1_15_IDs.i3* -gcd /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11057/01000-01999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz -t genie 
python /data/user/dpankova/double_pulse/Make_Images.py  -it 1 -o NuTAU_Make_Image_TEST -i /data/ana/Cscd/StartingEvents/NuGen_new/NuTau/medium_energy/IC86_flasher_p1=0.3_p2=0.0/l2/1/l2_0000000*.i3.zst -gcd /data/user/dpankova/double_pulse/GeoCalibDetectorStatus_2013.56429_V1_Modified.i3.gz -t genie 
#python /data/user/dpankova/double_pulse/Make_Images.py  -it 1 -o Genie_Make_image_TEST -i /data/ana/Cscd/StartingEvents/NuGen_new/NuTau/medium_energy/IC86_flasher_p1=0.3_p2=0.0/l2/1/l2_00000001.i3.zst -gcd /data/user/dpankova/double_pulse/GeoCalibDetectorStatus_2013.56429_V1_Modified.i3.gz -t genie
