#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import glob
import time
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors


# In[2]:


import tensorflow as tf
import os
# Set which GPU to use.  This probably needs to be done before any other CUDA vars get defined.
# Use the command "nvidia-smi" to get association of a particular GPU with a particular number.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]= "1,2,3,4"
from tensorflow.keras.models import load_model


# In[3]:


double_pulse_path = '/home/tmg5746/DoublePulse/'
data_path = '/data/tmg5746/'

nb_dir = os.path.join(double_pulse_path, 'stat_analysis')
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

from dtypes import save_atmos_dtype, preds_dtype, data_save_dtype,save_atmos_all_dtype,save_dtype,flux_dtype
from GaisserFlux import GaisserH4a


# In[5]:


#FluxErrorSystematics
#One sigma intervals of flux parameters
N_PRIM_CHILDREN = 3
STRINGS_TO_SAVE = 10
N_Y_BINS = 60
N_X_BINS = 500
N_CHANNELS = 3

INDEX_0 = 2.87
INDEX_0_u = 3.08
INDEX_0_l = 2.68
PHI_0 = 2.12
PHI_0_u = 2.61
PHI_0_l = 1.58

INDEX_1 = 2.50
INDEX_1_u = 2.41
INDEX_1_l = 2.59
PHI_1 = 2.23
PHI_1_u = 2.6
PHI_1_l = 1.83


INDEX_2 = 2.37
INDEX_2_u = 2.45
INDEX_2_l = 2.28
PHI_2 = 1.36
PHI_2_u = 1.6
PHI_2_l = 1.11


# In[6]:


# I had to change this in Make_Images.py and here for files in /data/sim/IceCube/2016/ and 2020/
from dtypes import id_dtype, st_info_dtype, particle_dtype, veto_dtype, hese_old_dtype, hese_dtype, flux_dtype

genie_weight_dtype2 = np.dtype(
        [
            ('PrimaryNeutrinoAzimuth',np.float32),
            ('TotalColumnDepthCGS',np.float32),
            ('MaxAzimuth',np.float32),
            ('SelectionWeight',np.float32),
            ('InIceNeutrinoEnergy',np.float32),
            ('PowerLawIndex',np.float32),
            ('TotalPrimaryWeight',np.float32),
            ('PrimaryNeutrinoZenith',np.float32),
            ('TotalWeight',np.float32),
            ('PropagationWeight',np.float32),
            ('NInIceNus',np.float32),
            ('TrueActiveLengthBefore',np.float32),
            ('TypeWeight',np.float32),
            ('PrimaryNeutrinoType',np.int64),
            ('RangeInMeter',np.float32),
            ('BjorkenY',np.float32),
            ('MinZenith',np.float32),
            ('InIceNeutrinoType',np.float32),
            ('CylinderRadius',np.float32),
            ('BjorkenX',np.float32),
            ('InteractionPositionWeight',np.float32),
            ('RangeInMWE',np.float32),
            ('InteractionColumnDepthCGS',np.float32),
            ('CylinderHeight',np.float32),
            ('SimMode', np.float32),
            ('InjectionCylinderRadius', np.float32),
            ('MinAzimuth',np.float32),
            ('TotalXsectionCGS',np.float32),
            ('InjectionOrigin_y',np.float32),
            ('OneWeightPerType',np.float32),
            ('ImpactParam',np.float32),
            ('InteractionTypeWeight',np.float32),
            ('InteractionType',np.float32),
            ('TrueActiveLengthAfter',np.float32),
            ('InjectionCylinderHeight', np.float32),
            ('MaxZenith',np.float32),
            ('InjectionOrigin_x',np.float32),
            ('InjectionOrigin_z',np.float32),
            ('InteractionXsectionCGS',np.float32),
            ('PrimaryNeutrinoEnergy',np.float32),
            ('DirectionWeight',np.float32),
            ('InjectionAreaCGS',np.float32),
            ('MinEnergyLog',np.float32),
            ('SolidAngle',np.float32),
            ('LengthInVolume',np.float32),
            ('NEvents',np.uint32),
            ('OneWeight',np.float32),
            ('MaxEnergyLog',np.float32),
            ('InteractionWeight',np.float32),
            ('EnergyLost',np.float32)
        ]
    )

save_dtype2 = np.dtype(
    [
        ("id", id_dtype),
        ("preds", preds_dtype),
        ("qtot", np.float32),
        ("qst", st_info_dtype, N_CHANNELS),
        ("primary", particle_dtype),
        ("prim_daughter", particle_dtype),
        ("logan_veto", veto_dtype),
        ("hese", hese_dtype),
        ("weight_dict", genie_weight_dtype2),
        ("weight_val_0",flux_dtype),
        ("weight_val_1",flux_dtype),
        ("weight_val_2",flux_dtype)
    ]
)


# In[22]:


#File names
#This is not very nesseary, just to keep track of all the option
data_types = ['BurnSample','MuonGun','genie','corsika','data']
nu_types = ['NuTau_1','NuTau_2','NuMu_1','NuMu_2','NuE_1','NuE_2','NuE_3'] # 1=CC, 2=NC, 3=GR
nu_types = ['NuTau_1','NuTau_2']
syst_types = ['', 'p0=0.0_p1=0.0_domeff=1.00', 'a+.10', 'a-.10', 's+.10', 's-.10', 'a+.05', 'a-.05', 's+.05', 's-.05', 'p0=0.0_p1=0.0_domeff=1.10', 'p0=0.0_p1=0.0_domeff=0.90', 'p0=1.0_p1=0.0_domeff=1.00', 'p0=-1.0_p1=0.0_domeff=1.00', 'p0=0.0_p1=0.2_domeff=1.00', 'p0=0.0_p1=-0.2_domeff=1.00']
syst_types = ['p0=1.0_p1=0.0_domeff=1.00', 'p0=-1.0_p1=0.0_domeff=1.00', 'p0=0.0_p1=0.2_domeff=1.00', 'p0=0.0_p1=-0.2_domeff=1.00']
#nu_type = nu_types[2]
data_type = data_types[2]
#syst = syst_types[0]

for syst in syst_types:
    for nu_type in nu_types:

        output_name = data_type +'_'+nu_type
        folder = 'nominal'
        if syst != '':
            output_name += '_'+syst
            folder = 'syst'
        print(output_name)


        # In[23]:


        #Weighting function for Nugen
        i3_per_npz = 1
        def get_rates_genie(one_weights, nu_E, n_npz_files, i3_per_npz, evts_per_i3file, spectral_index=INDEX_0, phi_0=PHI_0):
            ''' returns the per-year weights for the given input parameters '''
            total_events = n_npz_files*i3_per_npz*evts_per_i3file
            secs_per_year = 31536000
            flux_weights = 1e-18*phi_0*(nu_E/100e3)**(-spectral_index)
            return flux_weights/total_events*one_weights


        # In[24]:


        files_grabbed = glob.glob(os.path.join(data_path, 'Images/'+folder+'/'+nu_type[:-2]+'/Images_'+nu_type+'*'+syst+'*_data.npz'))
        print(os.path.join(data_path, 'Images/'+folder+'/'+nu_type[:-2]+'/Images_'+nu_type+'*'+syst+'*_data.npz'))

        # In[25]:


        #check what files you grabbed
        #print(files_grabbed)


        # In[26]:


        #count how many events you have in your image files
        #necessary for merging
        size =0
        for file_name in files_grabbed:
            x = np.load(file_name, mmap_mode="r")['arr_0']
            size = size +len(x)
        print(size)


        # In[27]:


        #load models
        model_1 = load_model(os.path.join(double_pulse_path, 'AC922/vgg16_200k_Qst_2000_2/vgg16_200k_QSt2000_dataset_norm_2.h5'))
        model_2 = load_model(os.path.join(double_pulse_path, 'AC922/vgg16_20k_Qst_2000_Corsika_20904/vgg16_20k_QSt2000_corsika_20904.h5'))
        model_3 = load_model(os.path.join(double_pulse_path, 'AC922/vgg16_700k_Qst_2000_MuvsTau_4/vgg16_700k_QSt2000_dataset_norm_MuVsTau_3.h5'))

        mean_1 = 0.0012322452384978533
        std_1  = 0.009694634936749935
        mean_2 = 0.00025147481937892735
        std_2 = 0.005774625577032566
        mean_3 = 0.00036459346301853657
        std_3  = 0.007035365793853998


        # In[ ]:


        #merge neutrino sim with flux systematics
        pos = 0
        print(output_name)
        weight_name = 'weight_dict'
        n_files = len(files_grabbed)
        start = time.time()
        data = np.lib.format.open_memmap(os.path.join(data_path, 'Scores/'+output_name+'.npy'), mode = 'w+', dtype =save_dtype2, shape=(size,))

        n_files_txt = open(os.path.join(data_path, 'Scores/'+output_name+'_n_files.txt'), 'w')
        n_files_txt.write(str(len(files_grabbed))+'\n')
        n_files_txt.close()

        for file_name in files_grabbed:
            x = np.load(file_name, mmap_mode="r")['arr_0']
            y = np.zeros(x.shape[0],dtype = save_dtype2)
            im = (x['image']-mean_1)/std_1
            pred_n1 = model_1.predict([im[:,0,:,:,:1],im[:,0,:,:,1:2],im[:,0,:,:,2:3]],batch_size =1)
            im = (x['image']-mean_2)/std_2
            pred_n2 = model_2.predict([im[:,0,:,:,:1],im[:,0,:,:,1:2],im[:,0,:,:,2:3]],batch_size =1)
            im = (x['image']-mean_3)/std_3
            pred_n3 = model_3.predict([im[:,0,:,:,:1],im[:,0,:,:,1:2],im[:,0,:,:,2:3]],batch_size =1)
            print(file_name, len(x), len(y), pos)

            for n,e in enumerate(x):

                preds = np.zeros(1,dtype = preds_dtype)
                preds[['n1','n2','n3']] = (pred_n1[n],pred_n2[n],pred_n3[n])

                weight_vals_0 = np.zeros(1,dtype = flux_dtype)
                weight_val_0 = get_rates_genie(e[weight_name]['OneWeight'], e[weight_name]['PrimaryNeutrinoEnergy'], n_npz_files= n_files,                                     i3_per_npz = i3_per_npz, evts_per_i3file = e[weight_name]["NEvents"],spectral_index=INDEX_0, phi_0=PHI_0)
                weight_val_0_nu = get_rates_genie(e[weight_name]['OneWeight'], e[weight_name]['PrimaryNeutrinoEnergy'], n_npz_files= n_files,                                     i3_per_npz = i3_per_npz, evts_per_i3file = e[weight_name]["NEvents"],spectral_index=INDEX_0, phi_0=PHI_0_u)
                weight_val_0_nl = get_rates_genie(e[weight_name]['OneWeight'], e[weight_name]['PrimaryNeutrinoEnergy'], n_npz_files= n_files,                                     i3_per_npz = i3_per_npz, evts_per_i3file = e[weight_name]["NEvents"],spectral_index=INDEX_0, phi_0=PHI_0_l)
                weight_val_0_su = get_rates_genie(e[weight_name]['OneWeight'], e[weight_name]['PrimaryNeutrinoEnergy'], n_npz_files= n_files,                                     i3_per_npz = i3_per_npz, evts_per_i3file = e[weight_name]["NEvents"],spectral_index=INDEX_0_u, phi_0=PHI_0)
                weight_val_0_sl = get_rates_genie(e[weight_name]['OneWeight'], e[weight_name]['PrimaryNeutrinoEnergy'], n_npz_files= n_files,                                     i3_per_npz = i3_per_npz, evts_per_i3file = e[weight_name]["NEvents"],spectral_index=INDEX_0_l, phi_0=PHI_0)
                weight_vals_0[['nom','nu','nl','su','sl']] = (weight_val_0,weight_val_0_nu,weight_val_0_nl,weight_val_0_su,weight_val_0_sl)

                weight_vals_1 = np.zeros(1,dtype = flux_dtype)
                weight_val_1 = get_rates_genie(e[weight_name]['OneWeight'], e[weight_name]['PrimaryNeutrinoEnergy'], n_npz_files= n_files,                                       i3_per_npz = i3_per_npz, evts_per_i3file = e[weight_name]["NEvents"],spectral_index=INDEX_1, phi_0=PHI_1)
                weight_val_1_nu = get_rates_genie(e[weight_name]['OneWeight'], e[weight_name]['PrimaryNeutrinoEnergy'], n_npz_files= n_files,                                       i3_per_npz = i3_per_npz, evts_per_i3file = e[weight_name]["NEvents"],spectral_index=INDEX_1, phi_0=PHI_1_u)
                weight_val_1_nl = get_rates_genie(e[weight_name]['OneWeight'], e[weight_name]['PrimaryNeutrinoEnergy'], n_npz_files= n_files,                                       i3_per_npz = i3_per_npz, evts_per_i3file = e[weight_name]["NEvents"],spectral_index=INDEX_1, phi_0=PHI_1_l)
                weight_val_1_su = get_rates_genie(e[weight_name]['OneWeight'], e[weight_name]['PrimaryNeutrinoEnergy'], n_npz_files= n_files,                                       i3_per_npz = i3_per_npz, evts_per_i3file = e[weight_name]["NEvents"],spectral_index=INDEX_1_u, phi_0=PHI_1)
                weight_val_1_sl = get_rates_genie(e[weight_name]['OneWeight'], e[weight_name]['PrimaryNeutrinoEnergy'], n_npz_files= n_files,                                       i3_per_npz = i3_per_npz, evts_per_i3file = e[weight_name]["NEvents"],spectral_index=INDEX_1_l, phi_0=PHI_1)
                weight_vals_1[['nom','nu','nl','su','sl']] = (weight_val_1,weight_val_1_nu,weight_val_1_nl,weight_val_1_su,weight_val_1_sl)

                weight_vals_2 = np.zeros(1,dtype = flux_dtype)
                weight_val_2 = get_rates_genie(e[weight_name]['OneWeight'], e[weight_name]['PrimaryNeutrinoEnergy'], n_npz_files= n_files,                                        i3_per_npz = i3_per_npz, evts_per_i3file = e[weight_name]["NEvents"],spectral_index=INDEX_2, phi_0=PHI_2)
                weight_val_2_nu = get_rates_genie(e[weight_name]['OneWeight'], e[weight_name]['PrimaryNeutrinoEnergy'], n_npz_files= n_files,                                        i3_per_npz = i3_per_npz, evts_per_i3file = e[weight_name]["NEvents"],spectral_index=INDEX_2, phi_0=PHI_2_u)
                weight_val_2_nl = get_rates_genie(e[weight_name]['OneWeight'], e[weight_name]['PrimaryNeutrinoEnergy'], n_npz_files= n_files,                                        i3_per_npz = i3_per_npz, evts_per_i3file = e[weight_name]["NEvents"],spectral_index=INDEX_2, phi_0=PHI_2_l)
                weight_val_2_su = get_rates_genie(e[weight_name]['OneWeight'], e[weight_name]['PrimaryNeutrinoEnergy'], n_npz_files= n_files,                                        i3_per_npz = i3_per_npz, evts_per_i3file = e[weight_name]["NEvents"],spectral_index=INDEX_2_u, phi_0=PHI_2)
                weight_val_2_sl = get_rates_genie(e[weight_name]['OneWeight'], e[weight_name]['PrimaryNeutrinoEnergy'], n_npz_files= n_files,                                        i3_per_npz = i3_per_npz, evts_per_i3file = e[weight_name]["NEvents"],spectral_index=INDEX_2_l, phi_0=PHI_2)
                weight_vals_2[['nom','nu','nl','su','sl']] = (weight_val_2,weight_val_2_nu,weight_val_2_nl,weight_val_2_su,weight_val_2_sl)

                y[["id","preds","weight_val_0","weight_val_1","weight_val_2","qtot","qst","primary","prim_daughter","logan_veto","hese","weight_dict",]][n]=        (e['id'],preds,weight_vals_0,weight_vals_1,weight_vals_2,e['qtot'],e['qst'],e['primary'],e['prim_daughter'],e['logan_veto'],e['hese'],e[weight_name])
            #print("\rPercent = "+str(round(n/x.shape[0]*100,3))+" "+str(n)+" of "+str(x.shape[0])+\
            #' Total = '+str(round((pos+n)/size*100,3))+" "+str(pos+n)+" of "+str(size), end="")
            data[pos:pos+len(x)] = y
            pos = pos + len(x)

        end = time.time()
        print(end - start)


