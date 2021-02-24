This folder contains some remains of our first attempts to use machine learning on icecube waveforms
THEY ARE VIEW ONLY, since data no longer exists you cant ran them or pictures will disappear

Clustering: attemp at using clustering to separate double pulses

Network training:
DL_parallel_trial: one of the first attempts at using CNNs with images from a sinlge string
DL_checks: an attempt to see what network is learing, visualising the intermediate layers
DL_heatmap: produces a map wich shows that parts of image were important  to making a final decision

Old_format_data_file_test: If anyone needs to know what old image format data looked like, this is it:
two files, same length. One has images, other event information.

Old_format_file_merging: we merged all the files inot one giant memory mapped file

KDE_smoth: trying to smooth the images with kerned density estimation, in hopes of improving something

DP_model_11c.h5 - one surviving early model, need 300x60 images for the inputs (no data!)

Presentation plots: really old images I still use im my presentaion. No data, So if you rerun they would have ot be remade (shame on me, i know) 
(NuESig and NuTauSig are images as pdfs)

TransferLearning: an attempt to use pre-trained convolutional part of vgg16 as is (without training it), didn't work out
