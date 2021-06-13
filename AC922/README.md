Training Networks\
=================
\
\
./cnn_common_original/ - original Aaron's code for Network training\
\
./cnn_common/ - most recent version of Aaron's code\
`taucnn_models.py` contains the model architectures.\
\
\
vgg16_200k_Qst_2000_2/ - NET1 NuE vs NuTau Network\
vgg16_700k_Qst_2000_MuvsTau_3/ - NET3 NuMu vs NuTau Network\
vgg16_30k_Qst_2000_Corsika_3/ - NET2 Corsika vs Nutau\
vgg16_20k_Qst_2000_Corsika_20904/ - newer NET2 without oversizing\
\
To train a new Network:

- Copy one of the three folders above. 
- Delete logs, .txt .pdf and .h5 files.
- Set the training parameters in `train_vgg16...py` 
- Make sure all the paths are set correctly (to the data, to the scripts)
- `python train_vgg16....py`
