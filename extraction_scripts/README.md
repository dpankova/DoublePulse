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

# Step by step
Previous part was written by Dasha, this part is written by Timothee.
## Produce the images
In `/home/tgregoire/DoublePulse/extraction_scripts/` use `condor_submit` on the Make_Images_*.sub, it will run Make_Images.py into jobs.
Each Make_Images_*.sub corresponds to different sets, the documentation about some of them is in https://wiki.icecube.wisc.edu/index.php/Simulation_and_Datas
ets_Used

The .log, .out and .error files of the jobs are in `/scratch/tgregoire/DoublePulse/` (accessible from submit-1 not from cobalt). If some files are corrupted
it will kill the job, check the error file for that and add the corrupted files in the corrupted_files list of Make_Images.py before re-running the job.
The output files are in /data/user/tgregoire/DoublePulse/.

## Transfer the image files to luzern
Connect to luzern, go to `/data/tmg5746/Images/` and use rsync.
For example, for nominal NuE files: `rsync --update -r --include='*NuE*' tgregoire@data.icecube.wisc.edu:"/data/user/tgregoire/DoublePulse/nominal/NuE/" ./no
minal/NuE/`

## Process the images to get the scores
In `DoublePulse/notebooks/tgregoire/` modify `ProcessImages.py` line 159-169 to process only the files you are interested in (neutrino type, nom
inal or which systematic set...) and run it.
The output score files are in `/data/tmg5746/Scores/`


 
