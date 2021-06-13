How to make images:\
\
`Make_Images.py` is main script for image creation:
- -i - Input I3 files
- -o - Output file
- -gcd - GCD I3 file
- -t - Data type (corsika, data, muongun, genie)
- -y - Simulation production year (only matters for coriska)
- -set - Data set number (only matters for corsika)
- -it - Interaction type (CC,NC,GR only matter for genie)
 
It produces `.txt` and `.npz` files.\ 
`.txt` files contain I3 files name and number of events produced for each of them.\ 
Script `Summarize_text_files.py` can provide a condenced information from `.txt` files.\
\
`.txt` files contain images and other information about the event.\
(Check dtypes in `Make_Images.py` to see what exactly they contain.)\
\
`Make_Images.sh` is a bash script for running tests on cobalt fo every data type.\
To use uncomemnt the relevant line.\
\
`.sub` are the files for running jobs on NPX cluster for each data type.\
\
`Make_Dags.py` - create `.dag` files for running dagman jobs on NPX cluster.\
\
./BurnSample/ - extra scripts for working with real data, including burn sample.\
\
./IceProd/ - contains config.json and Make_Images.py scripts what only accepts one file at a time for running on IceProd.\
\
MAKE SURE TO SET THE DIRECTORIES BEFORE RUNNING


 
