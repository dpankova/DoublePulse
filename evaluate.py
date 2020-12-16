#!/usr/bin/env python
from __future__ import division
import argparse
import numpy as np
import glob

import Weighting

parser = argparse.ArgumentParser()

parser.add_argument("-i","--infile",
                    dest="infile",
                    type=str,
                    nargs = "+",
                    help="[I]nfile name")

parser.add_argument("-o","--outfile",
                    dest="outfile",
                    type=str,
                    default="Out",
                    help="base name for outfile")

parser.add_argument("-dt","--data_type",
                    dest="data_type",
                    type=str,
                    choices=['genie','corsika'],
                    help="corsika or genie")

parser.add_argument("-it","--interaction_type",
                    dest="int_type",
                    type=str,
                    default='1'
                    choices=['1','2','3'],
                    help="CC =1, NC=2, GR=3")

parser.add_argument("-nt","--nu_type",
                    dest="nu_type",
                    type=str,
                    default ='NuTau'
                    choices=['NuE','NuMu','NuTau'],
                    help="neutrino flavor")

parser.add_argument("-fdr","--folder_number",
                    dest="folder",
                    type=str,
                    default ='01',
                    help="numbers 01-13 for genie, something like 00000_00999 for corsika")

parser.add_argument("-f","--file_number",
                    dest="file_number",
                    type=str,
                    default='01',
                    help="01 to 99|01 to 09|*")

parser.add_argument("-ds","--dataset",
                    dest="dataset",
                    type=str,
                    default='1',
                    help="corsika dataset")

parser.add_argument("-n","--number_files",
                    dest="n_files",
                    type=str,
                    default='100',
                    help="number of L2 files total")


args=parser.parse_args()
infiles=args.infile
outfile=args.outfile
data_type=args.data_type
nu_type=args.nu_type
int_type=args.int_type
folder=args.folder
files=args.file_number

dataset=args.dataset
n_files=args.n_files


if  data_type = 'genie':
    if nu_type ="NuTau":
        i3_per_npz = 100 # 10 for NuE1 and NuMu1, 100 otherwise
    else:
        i3_per_npz = 10

name = outfile+'_'+nu_type+'_'+str(int_type)+'_'+str(folder)+'_'+str(files)
#name = outfile+'_'+nu_type+'_'+str(int_type)+'_'+str(folder)+'_'+str(files)+'?_'


print(name)

N_PRIM_CHILDREN = 3
STRINGS_TO_SAVE = 10
N_Y_BINS = 60
N_X_BINS = 500
N_CHANNELS = 3

DEFAULT_INDEX = 2.88
DEFAULT_PHI = 2.1467

outer_strings = set([1,2,3,4,5,6,7,13,14,21,22,30,31,40,41,50,51,59,60,67,68,72,73,74,75,76,77,78])

preds_dtype = np.dtype(
    [
        ('n1', np.float32),
        ('n2', np.float32),
        ('n3', np.float32),
        ('n4', np.float32)
    ]
)
st_info_dtype = np.dtype(
    [
        ('q', np.float32),
        ('num', np.uint32),
        ('dist', np.float32)
    ]
)
id_dtype = np.dtype(
    [
        ("run_id", np.uint32),
        ("sub_run_id", np.uint32),
        ("event_id", np.uint32),
        ("sub_event_id", np.uint32),
    ]
)
particle_dtype = np.dtype(
    [
        ("tree_id", np.uint32,(2)),
        ("pdg", np.int32),
        ("energy", np.float32),
        ("position", np.float32,(3)),
        ("direction", np.float32,(2)),
        ("time", np.float32),
        ("length", np.float32)
    ]
)
veto_dtype = np.dtype(
    [
        ("SPE_rlogl", np.float32),
        ("Cascade_rlogl", np.float32),
        ("SPE_rlogl_noDC", np.float32),
        ("Cascade_rlogl_noDC", np.float32),
        ("FirstHitZ", np.float32),
        ("VHESelfVetoVertexPosZ", np.float32),
        ("LeastDistanceToPolygon_Veto", np.float32)
    ]
)
hese_dtype = np.dtype(
        [
            ("qtot", np.float32),
            ("vheselfveto", np.bool_),
            ("vheselfvetovertexpos", np.float32,(3)),
            ("vheselfvetovertextime", np.float32),
            ("llhratio", np.float32)
        ]
)

#if data_type =='genie':
WEIGHT_KEY = "I3MCWeightDict"
weight_dtype = np.dtype(
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
        ('PrimaryNeutrinoType',np.float32),
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
        ('MinAzimuth',np.float32),
        ('TotalXsectionCGS',np.float32),
        ('OneWeightPerType',np.float32),
        ('ImpactParam',np.float32),
        ('InteractionType',np.float32),
        ('TrueActiveLengthAfter',np.float32),
        ('MaxZenith',np.float32),
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
info_dtype = np.dtype(
    [
        ("id", id_dtype),
        ("image", np.float32, (N_X_BINS, N_Y_BINS, N_CHANNELS)),
        ("qtot", np.float32),
        ("qst", st_info_dtype, N_CHANNELS),
        ("qst_all", st_info_dtype, STRINGS_TO_SAVE),
        ("primary", particle_dtype),
        ("prim_daughter", particle_dtype),
        ("primary_child_energy", np.float32,(N_PRIM_CHILDREN)),
        ("primary_child_pdg", np.float32,(N_PRIM_CHILDREN)),
        ("logan_veto", veto_dtype),
        ("hese_old", hese_dtype),
        ("hese", hese_dtype),
        ("weight", weight_dtype)
    ]
)
save_dtype = np.dtype(
    [
        ("id", id_dtype),
        ("preds", preds_dtype),
        ("weight_val", np.float32),
        ("qtot", np.float32),
        ("qst", st_info_dtype, N_CHANNELS),
        ("qst_all", st_info_dtype, STRINGS_TO_SAVE),
        ("primary", particle_dtype),
        ("prim_daughter", particle_dtype),
        ("hese", hese_dtype),
        ("weight", weight_dtype)
    ]
)
