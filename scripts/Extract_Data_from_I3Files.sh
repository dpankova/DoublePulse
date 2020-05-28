#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT /data/user/dpankova/oscNext/oscNext_meta/build/ 

python /data/user/dpankova/double_pulse/Extract_Data_from_I3Files.py  -o Corsika_Extract_TEST -i /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11057/00000-00999/Level2_IC86.2012_corsika.011057.000000.i3.bz2  -gcd /data/sim/IceCube/2012/filtered/level2/CORSIKA-in-ice/11057/01000-01999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz -t corsika

#python /data/user/dpankova/double_pulse/Extract_Data_from_I3Files.py -o Genie_Extract_TEST -i /data/ana/Cscd/StartingEvents/NuGen_new/NuTau/medium_energy/IC86_flasher_p1=0.3_p2=0.0/l2/1/l2_00000001.i3.zst -gcd /data/user/dpankova/double_pulse/GeoCalibDetectorStatus_2013.56429_V1_Modified.i3.gz -t genie
