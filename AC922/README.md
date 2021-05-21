Triaining Networks

./cnn_common_original/ - original Aaron's code for NEtwork training
./cnn_common/ - most recent version of Aaron's code

./vgg16_200k_Qst_2000_2/ - NET1 NuE vs NuTau Network
./vgg16_700k_Qst_2000_MuvsTau_3/ - NET3 NuMu vs NuTau Network
./vgg16_30k_Qst_2000_Corsika_3/ - NET2 Corsika vs Nutau

To train a new Network copy one of the three folders above, delete logs, .txt .pdf and .h5 files.
Set the training parameters in train_vgg16...py (Make sure to set the paths to the data)
do: python train... .py
