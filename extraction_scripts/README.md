How to make images:

Make_Images.py - main script for image creation:
 -i - Input I3 files
 -o - Output file
 -gcd - GCD I3 file
 -t - Data type (corsika, data, muongun, genie)
 -y - Simulation production year (only matters for coriska)
 -set - Data set number (only matters for corsika)
 -it - Interaction type (CC,NC,GR only matter for genie)
Produces .txt and .npz files. TXT files contain I3 files name and number of events produced for each of them.
NPZ files contain images and other informatin about the event.

Make_Images.sh - bash script for running tests on cobalt fo every data type

Sub files for each type of data to run on NPX cluster

Make_Dags.py - create .dag files for NPX cluster

./BurnSample/ - extra files for working with real data
./IceProd/ - contains config.json and Make_Images.py scripts what only accepts one file at a time for running on IceProd.

MAKE SURE TO SET THE DIRECTORIES BEFORE RUNNING


 
