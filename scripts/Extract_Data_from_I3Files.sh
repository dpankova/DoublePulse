#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT /data/user/dpankova/oscNext/oscNext_meta/build/ 

#echo "hello"
#python /data/user/dpankova/double_pulse/100TeVParticles.py 

#python /data/user/dpankova/double_pulse/Make_NewFormat_All_Weights.py  -p 11 -o LostEventsTest -i /data/user/dpankova/double_pulse/Lost_events.i3.bz2
python /data/user/dpankova/double_pulse/Make_NewFormat_All_MCTree.py  -o NewFormat_NuTau_test  -i /data/ana/Cscd/StartingEvents/NuGen_new/NuTau/medium_energy/IC86_flasher_p1=0.3_p2=0.0/l2/1/l2_00000001.i3.zst
