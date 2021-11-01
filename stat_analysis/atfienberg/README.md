Instructions
============

Take the following steps to run these notebooks on the cobalts:

1. Install [conda](https://docs.conda.io/en/latest/miniconda.html) if it is not already installed. Either follow the instructions on the conda website or simply run `bash /data/user/afienberg/conda_installer/Miniconda3-py37_4.10.3-Linux-x86_64.sh`. I recommend choosing **not** to run `conda init` during the initial installation.

2. Activate conda by running `eval "$($CONDA_DIR/bin/conda shell.bash hook)"`. If `CONDA_DIR` is not defined, instead substitute the path to your conda installation.

3. Navigate to this directory (the one containing this README file). Execute the command `conda env create -f cobalt_environment.yml`. This will create a new conda environment called `doublepulse` with all the dependencies necessary to run the analysis notebooks in this directory. 

4. Activate the `doublepulse` environment with `conda activate doublepulse`.

5. Launch jupyter, e.g. `jupyter-lab --no-browser --port <portnum>`. You will need to create an SSH tunnel to be able to access this jupyter server from your local browser. An example command to create an SSH tunnel from your local machine through `pub` to `cobalt08` is `ssh -t -t -L<port>:localhost:<port> pub 'ssh -L<port>:localhost:<port> cobalt08`.

6. Run the notebooks. The outputs should be identical to those shown in versions on GitHub. 
