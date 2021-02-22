#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT /data/user/dpankova/oscNext/oscNext_meta/build/

import icecube
from I3Tray import *
<<<<<<< HEAD
from icecube.common_variables import hit_statistics
=======
>>>>>>> master
from icecube import icetray, dataclasses, dataio, WaveCalibrator, common_variables
from icecube.recclasses import I3DipoleFitParams
from icecube.recclasses import I3LineFitParams
from icecube.recclasses import I3CLastFitParams
from icecube.recclasses import I3CscdLlhFitParams
from icecube.recclasses import I3TensorOfInertiaFitParams
from icecube.recclasses import I3FillRatioInfo
from icecube.recclasses import CramerRaoParams
from icecube.recclasses import I3StartStopParams
from icecube.gulliver import I3LogLikelihoodFitParams
<<<<<<< HEAD
from icecube.weighting.fluxes import GaisserH4a
from icecube.weighting.weighting import from_simprod
from icecube import tensor_of_inertia
=======
>>>>>>> master

import Reconstruction
import PolygonContainment 
import QTot 

import numpy as np
import glob
import sys
import argparse


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

parser.add_argument("-gcd","--gcdfile",
                    dest="gcdfile",
                    type=str,
                    help="gcdfile location")

parser.add_argument("-t","--data_type",
                    dest="data_type",
                    type=str,
		    choices=['genie','corsika'],
                    help="corsika or genie")

<<<<<<< HEAD
parser.add_argument("-set","--dataset",
                    dest="dataset",
                    type=int,
		    default=-1,
                    help="corsika dataset number")

=======
>>>>>>> master
parser.add_argument('-e1', '--min_energy',
                    dest='energy_min',
                    type=int,
                    default=500000,
                    help='minimum energy of primary meson in GeV')

parser.add_argument('-it', '--interaction_type',
                    dest='it',
                    type=int,
                    default=1,
                    help='Interaction types are : CC -1, NC -2 ,GR-3')

parser.add_argument('-e2', '--max_energy',
                    dest='energy_max',
                    type=int,
                    default=1500000,
                    help= 'maximum energy of primary meson in GeV')

parser.add_argument("-sk","--skip_files",
                    dest="skip",
                    type=str,
		    default =["l2_00001014", "011057.001312","011057.003278","011057.004247","011057.005703","011057.008446", "011057.004247","011057.003278","011057.005703","011057.001312","011057.003278","011057.004247","011057.005703"],
                    nargs="+",
                    help="skip files with that srting in the name")


args = parser.parse_args()
infiles=args.infile
outfile=args.outfile
gfile=args.gcdfile
it=args.it
<<<<<<< HEAD
dataset=args.dataset
=======
>>>>>>> master
data_type=args.data_type
energy_min=args.energy_min
energy_max=args.energy_max
skip=args.skip
print "skip", skip
<<<<<<< HEAD
print(infiles)
=======

>>>>>>> master

#PARAMETERS
GEO = 'ic86'
C=0.299792458                                                                  
N=1.3195
V=C/N
QST_THRES = 400 # PE, cut on max string charge
QTOT_THRES = 1000 #PE, cut on total charge
DIST_ST_CUT = 150**2 #m, look at string within this distance of the max string
N_PRIM_CHILDREN = 3 #number of daughter particles to get from MCTree Primary


#image size
STRINGS_TO_SAVE = 10
N_Y_BINS = 60
N_X_BINS = 500
N_CHANNELS = 3


#ADD GCD to the start of the input filelist
file_list = []
geofile = dataio.I3File(gfile)
i_frame = geofile.pop_frame()
g_frame = geofile.pop_frame()
<<<<<<< HEAD
geofull = g_frame["I3Geometry"]
=======
>>>>>>> master
geometry = g_frame["I3Geometry"].omgeo
file_list.append(gfile)

#Add input files to file list and skip bad files
for files in infiles:
<<<<<<< HEAD
    print(files)
    for filename in glob.glob(files):
        print(filename)
        skip_it = False
        for sk in skip:
	    if (sk in filename):
	        skip_it = True
        if not skip_it:
	    file_list.append(filename)
=======
    for filename in glob.glob(files):
        skip_it = False
        for sk in skip:
            skip_it = sk in filename
        if not skip_it:
            file_list.append(filename)
>>>>>>> master

print(file_list) #Final file list


#Define structured types
st_info_dtype = np.dtype(
<<<<<<< HEAD
    [                                                                                       
        ('q', np.float32),
=======
    [                                                                                                                                        ('q', np.float32),
>>>>>>> master
        ('num', np.uint32),
        ('dist', np.float32)
    ]
)
<<<<<<< HEAD

=======
>>>>>>> master
id_dtype = np.dtype(
    [
        ("run_id", np.uint32),
        ("sub_run_id", np.uint32),
        ("event_id", np.uint32),
        ("sub_event_id", np.uint32),
    ]
)
<<<<<<< HEAD

map_dtype = np.dtype(
    [                                                                                       
	("id", id_dtype),
        ('raw', np.int32),
        ('st_raw', np.int32,(3)),
        ('pulses', np.int32),
        ('st_pulses', np.int32,(3)),
	('cal', np.int32),
        ('st_cal', np.int32,(3)),
	('hlc', np.int32),
        ('st_hlc', np.int32,(3)),
	('slc', np.int32),
        ('st_slc', np.int32,(3))
    ]
)

=======
>>>>>>> master
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
<<<<<<< HEAD
       
=======
>>>>>>> master
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
if data_type =='genie':
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
else: #if data_type =='corsika'
    WEIGHT_KEY = "CorsikaWeightMap"
    weight_dtype = np.dtype(
	[
            ("AreaSum" ,np.float32),
            ("Atmosphere",np.float32),
            ("CylinderLength",np.float32),
            ("CylinderRadius" ,np.float32),
            ("DiplopiaWeight",np.float32),
            ("EnergyPrimaryMax",np.float32),
            ("EnergyPrimaryMin",np.float32),
            ("FluxSum",np.float32),
            ("Multiplicity",np.float32),
            ("NEvents",np.float32),
            ("OldWeight",np.float32),
            ("OverSampling",np.float32),
            ("Polygonato",np.float32),
            ("PrimaryEnergy",np.float32),
            ("PrimarySpectralIndex",np.float32),
            ("PrimaryType",np.float32),
            ("ThetaMax",np.float32),
            ("ThetaMin" ,np.float32),
            ("TimeScale",np.float32),
            ("Weight",np.float32)
	]                            
    )    
info_dtype = np.dtype(
    [
        ("id", id_dtype),
        ("image", np.float32, (N_X_BINS, N_Y_BINS, N_CHANNELS)),
        ("qtot", np.float32),
<<<<<<< HEAD
	("qoth", np.float32),
	("cog", np.float32,(3)),
	("moi", np.float32),
	("ti", np.float32,(4)),
        ("qst", st_info_dtype, N_CHANNELS),
	("qst_all", st_info_dtype, STRINGS_TO_SAVE),
	("map", map_dtype),
=======
        ("qst", st_info_dtype, N_CHANNELS),
>>>>>>> master
	("primary", particle_dtype),
        ("prim_daughter", particle_dtype),
        ("primary_child_energy", np.float32,(N_PRIM_CHILDREN)),
	("primary_child_pdg", np.float32,(N_PRIM_CHILDREN)),
        ("logan_veto", veto_dtype),                                                  
	("hese_old", hese_dtype),                                                  
<<<<<<< HEAD
	("hese", hese_dtype),                                                               
	("weight", weight_dtype)
	
=======
	("hese", hese_dtype),                                                                                                    
	("weight", weight_dtype)
>>>>>>> master

    ]
)


#main data array
data = []
<<<<<<< HEAD
temp = []
qtot = 0 #total event charge
st_info_all = np.zeros(STRINGS_TO_SAVE,dtype = st_info_dtype) #[Qst, Nst, DistST] 
=======
qtot = 0 #total event charge
>>>>>>> master
st_info = np.zeros(N_CHANNELS,dtype = st_info_dtype) #[Qst, Nst, DistST] for each image

#Remove events that don't have the required info
def Check_Data(frame):
    #HEADER and STREAM
<<<<<<< HEAD
    
=======
>>>>>>> master
    has_header = frame.Has("I3EventHeader")
    has_stream = frame["I3EventHeader"].sub_event_stream != "NullSplit"
    if not has_stream:
	return False #Dont use NullSplit frames
   
    has_rawdata =  False
    if frame.Has("InIceRawData"):
	try:
	    raw_data = 0
	    raw_data = frame["InIceRawData"]
	    if len(raw_data) != 0:
		has_rawdata = True
	    else: 
		has_rawdata = False
	except:
	    has_rawdata = False

    has_pulses = False
    if frame.Has("SplitInIcePulses"):
        try:
            pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulses')    
	    if not (len(pulses) == 0):
                has_pulses = True
        except:
            has_pulses = False

    has_mctree = frame.Has("I3MCTree")
    has_weights = frame.Has(WEIGHT_KEY)

    #Images can be made if event has all the keys, passed == True
    passed = has_header and has_weights and has_rawdata and has_mctree and has_pulses
    if passed:
	if data_type == 'genie': #Keep only events with right interaction type
	    if frame[WEIGHT_KEY]['InteractionType'] != it:
		return False

      	return True
    else:
	return False


def Get_Charges(frame):
    global qtot
    global st_info
<<<<<<< HEAD
    global st_info_all

    #Calculate charges for cuts
    pulses= dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulses')
    #wf_map = frame["CalibratedWaveformsHLCATWD"]
=======

    #Calculate charges for cuts
    pulses= dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulses')
>>>>>>> master
    omkeys = pulses.keys()
    strings_set = set()
    string_keys = {}

    #Make a set of all hit strings and a dictinary of doms on those string
    for omkey in omkeys:                                               
        if omkey.string not in strings_set:
            strings_set.add(omkey.string)
            string_keys[omkey.string] = [omkey]
	else:
            string_keys[omkey.string].append(omkey)

    #Caculate charge of each string
    string_qs = []
    for string, doms in string_keys.items():
        string_q = 0
	for omkey in doms:
            qs = pulses[omkey]
            qdom = sum(i.charge for i in qs)
	    string_q = string_q + qdom

	# get associated string x-y position
        st_pos = geometry[doms[0]].position
        st_xy = np.array([st_pos.x, st_pos.y])     
	string_qs.append([string_q,string,st_xy])
        
    #sort strings by charge and find max
    string_qs.sort(key=lambda x: x[0], reverse = True)
    qtot = sum([i[0] for i in string_qs])
    max_qst = string_qs[0][0] 

    #MAKE Charge CUT
    if (max_qst < QST_THRES) or (qtot < QTOT_THRES):
	return False

<<<<<<< HEAD
    for st in range(STRINGS_TO_SAVE): 
        dist_st = np.sum((string_qs[0][2]-string_qs[st][2])**2)
        st_info_all[['q','num','dist']][st] = (string_qs[st][0],string_qs[st][1],dist_st)
    
=======
>>>>>>> master
    # find neighboring strings and sort by charge
    # include max charge string in this list
    # strings 11 and 19 are neighboring but almost 150 m apart
    
    near_max_strings = []
    for q, st, xy in string_qs:
        dist_st = np.sum((string_qs[0][2]-xy)**2)
        if dist_st < DIST_ST_CUT:
            near_max_strings.append((q, st, dist_st))

    if len(near_max_strings) < N_CHANNELS:
	return False

    for ch in range(N_CHANNELS): 
        st_info[['q','num','dist']][ch] = (near_max_strings[ch][0],near_max_strings[ch][1],near_max_strings[ch][2])
<<<<<<< HEAD
	
=======
    
>>>>>>> master
    return True

def Make_Image(frame):
    global data 
    global geometry
    global st_info

    #Log id info 
    id = np.zeros(1,dtype = id_dtype)
    H = frame["I3EventHeader"]
    id[["run_id","sub_run_id","event_id","sub_event_id"]] = (H.run_id,H.sub_run_id,H.event_id,H.sub_event_id)


    #Log Weight info
    weight = np.zeros(1,dtype = weight_dtype)
    w = dict(frame[WEIGHT_KEY])
    weight[list(w.keys())] = tuple(w.values())

    
    #Log MCTree info
    primary = np.zeros(1,dtype = particle_dtype)
    prim_daughter = np.zeros(1,dtype = particle_dtype)
    energy = 0
    pdg = 0
    
    mctree = frame["I3MCTree"]
    if data_type == 'genie':
	prim = dataclasses.get_most_energetic_neutrino(mctree)
    else:
	prim = dataclasses.get_most_energetic_muon(mctree)
	
    prim_chldn = mctree.children(prim.id)
    pdgs = []
    daughters = []
    energies = []
        
    for part in prim_chldn:
        energies.append(part.energy)
	daughters.append(part)
	pdgs.append(part.pdg_encoding)

    if len(energies) == 0:
<<<<<<< HEAD
	print("MCTree has no primary children")
	energies = np.zeros(N_PRIM_CHILDREN)
	pdgs = np.zeros(N_PRIM_CHILDREN)    
	primary[["tree_id","pdg","energy","position","direction","time","length"]] = ([prim.id.majorID, prim.id.minorID], prim.pdg_encoding, prim.energy,[prim.pos.x,prim.pos.y,prim.pos.z],[prim.dir.zenith,prim.dir.azimuth],prim.time, prim.length)
=======
	has_mctree = False
	print("MCTree has no primary children")
	energies = np.zeros(N_PRIM_CHILDREN)
	pdgs = np.zeros(N_PRIM_CHILDREN)    

>>>>>>> master
    else:
	zipped = zip(energies,pdgs,daughters)
	zipped_sort = sorted(zipped, key = lambda x: x[0], reverse =True)
	zipped_sort = np.array(zipped_sort)
	energies = np.zeros(N_PRIM_CHILDREN)
	pdgs = np.zeros(N_PRIM_CHILDREN)

	n_to_copy = len(zipped_sort)
	if n_to_copy > N_PRIM_CHILDREN:
            n_to_copy = N_PRIM_CHILDREN

	energies[:n_to_copy] = zipped_sort[:,0][:n_to_copy]
	pdgs[:n_to_copy] = zipped_sort[:,1][:n_to_copy]

<<<<<<< HEAD
	daughter = zipped_sort[0][2]
    
	primary[["tree_id","pdg","energy","position","direction","time","length"]] = ([prim.id.majorID, prim.id.minorID], prim.pdg_encoding, prim.energy,[prim.pos.x,prim.pos.y,prim.pos.z],[prim.dir.zenith,prim.dir.azimuth],prim.time, prim.length)
	prim_daughter[["tree_id","pdg","energy","position","direction","time","length"]] = ([daughter.id.majorID, daughter.id.minorID], daughter.pdg_encoding,daughter.energy,[daughter.pos.x,daughter.pos.y,daughter.pos.z],[daughter.dir.zenith,daughter.dir.azimuth],daughter.time,daughter.length)
=======
    daughter = zipped_sort[0][2]
    
    primary[["tree_id","pdg","energy","position","direction","time","length"]] = ([prim.id.majorID, prim.id.minorID], prim.pdg_encoding, prim.energy,[prim.pos.x,prim.pos.y,prim.pos.z],[prim.dir.zenith,prim.dir.azimuth],prim.time, prim.length)
    prim_daughter[["tree_id","pdg","energy","position","direction","time","length"]] = ([daughter.id.majorID, daughter.id.minorID], daughter.pdg_encoding,daughter.energy,[daughter.pos.x,daughter.pos.y,daughter.pos.z],[daughter.dir.zenith,daughter.dir.azimuth],daughter.time,daughter.length)
>>>>>>> master

 
       
    #Log HESE
    hese = np.zeros(2,dtype = hese_dtype)
    hese_qtot = 0 
    hese_vheselfveto = True
    hese_llhratio = 0
    hese_pos =[-9999,-9999,-9999]
    hese_time = 99999
    if frame.Has("HESE_VHESelfVeto"):
	hese_qtot = frame["HESE_CausalQTot"].value
	hese_vheselfveto = frame["HESE_VHESelfVeto"].value
	hese_llhratio = frame["HESE_llhratio"].value
	hese_pos = [frame["HESE_VHESelfVetoVertexPos"].x,frame["HESE_VHESelfVetoVertexPos"].y,frame["HESE_VHESelfVetoVertexPos"].z]
	hese_time = frame["HESE_VHESelfVetoVertexTime"].value
    hese[["qtot","vheselfveto","llhratio","vheselfvetovertexpos","vheselfvetovertextime"]][0] = (hese_qtot,hese_vheselfveto,hese_llhratio,hese_pos,hese_time) 

    hese_vheselfveto = True
    hese_pos =[-9999,-9999,-9999]
    hese_time = 99999
    if frame.Has("HESE2_VHESelfVeto"):
	hese_vheselfveto = frame["HESE2_VHESelfVeto"].value
	hese_pos = [frame["HESE2_VHESelfVetoVertexPos"].x,frame["HESE2_VHESelfVetoVertexPos"].y,frame["HESE2_VHESelfVetoVertexPos"].z]
	hese_time = frame["HESE2_VHESelfVetoVertexTime"].value
    hese[["qtot","vheselfveto","llhratio","vheselfvetovertexpos","vheselfvetovertextime"]][1] = (hese_qtot,hese_vheselfveto,hese_llhratio,hese_pos,hese_time) 

   
    #Log logan veto
    veto = np.zeros(1,dtype = veto_dtype)
    veto_cas_rlogl = 999
    veto_spe_rlogl = -999
    veto_cas_rlogl_ndc = 999
    veto_spe_rlogl_ndc = -999
    veto_fh_z = 999
    veto_svv_z = 999
    veto_ldp = -999
<<<<<<< HEAD
    
    if frame.Has('HESE3_VHESelfVetoVertexPos') and frame.Has('SPEFit32_DPFitParams') and frame.Has('CascadeLlhVertexFit_DPParams') and frame.Has('SPEFit32_noDC_DPFitParams') and frame.Has('CascadeLlhVertexFit_noDC_DPParams') and frame.Has('depthFirstHit') and frame.Has("LeastDistanceToPolygon_Veto"):
	 
=======
    if frame.Has('VHESelfVetoVertexPos') and frame.Has('SPEFit32_DPFitParams') and frame.Has('CascadeLlhVertexFit_DPParams') and frame.Has('SPEFit32_noDC_DPFitParams') and frame.Has('CascadeLlhVertexFit_noDC_DPParams') and frame.Has('depthFirstHit') and frame.Has("LeastDistanceToPolygon_Veto"):
>>>>>>> master
          veto_cas_rlogl = frame['CascadeLlhVertexFit_DPParams'].ReducedLlh
	  veto_spe_rlogl = frame['SPEFit32_DPFitParams'].rlogl
	  veto_cas_rlogl_ndc = frame['CascadeLlhVertexFit_noDC_DPParams'].ReducedLlh
	  veto_spe_rlogl_ndc = frame['SPEFit32_noDC_DPFitParams'].rlogl
	  veto_fh_z = frame['depthFirstHit'].value
<<<<<<< HEAD
	  veto_svv_z = frame['HESE3_VHESelfVetoVertexPos'].z
=======
	  veto_svv_z = frame['VHESelfVetoVertexPos'].z
>>>>>>> master
	  veto_ldp = frame["LeastDistanceToPolygon_Veto"].value


    veto[["SPE_rlogl","Cascade_rlogl","SPE_rlogl_noDC", "Cascade_rlogl_noDC","FirstHitZ","VHESelfVetoVertexPosZ","LeastDistanceToPolygon_Veto"]] = (veto_spe_rlogl,veto_cas_rlogl,veto_spe_rlogl_ndc,veto_cas_rlogl_ndc,veto_fh_z,veto_svv_z,veto_ldp)                    



    #make image from raw waveforms
    pulses= dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulses')
    wf_map = frame["CalibratedWaveformsHLCATWD"]
<<<<<<< HEAD
    wf_map_slc = frame["CalibratedWaveformsSLC"]
    wf_map_cal = frame["CalibratedWaveforms"]
    rs = frame["InIceRawData"]
#    for i in wf_map_slc.keys():
#        for j in wf_map_slc[i]:
#            print(i, j.waveform)
    in_hlc = [0,0,0]
    in_slc = [0,0,0]
    in_rs = [0,0,0]
    in_pls = [0,0,0]
    in_cal = [0,0,0]
    emaps = np.zeros(1,dtype = map_dtype)    
    for img_ch, (q, stnum, dist) in enumerate(st_info):
        for omkey in wf_map.keys():
	    if (omkey.string == stnum):
		in_hlc[img_ch] += 1
	for omkey in wf_map_slc.keys():
	    if (omkey.string == stnum):
		in_slc[img_ch] += 1
	for omkey in pulses.keys():
	    if (omkey.string == stnum):
		in_pls[img_ch] += 1
	for omkey in wf_map_cal.keys():
	    if (omkey.string == stnum):
		in_cal[img_ch] += 1
	for omkey in rs.keys():
	    if (omkey.string == stnum):
		in_rs[img_ch] += 1

    emaps[["id",'raw','st_raw','pulses','st_pulses','cal','st_cal','hlc','st_hlc','slc','st_slc']]=\
    (id[0],len(rs),in_rs,len(pulses),in_pls,len(wf_map_cal),in_cal,len(wf_map),in_hlc,len(wf_map_slc),in_slc)
    cogPos = hit_statistics.calculate_cog(geofull,frame["SplitInIcePulses"].apply(frame))     
#    print(emaps,cogPos)
 #   temp.append(emaps)
    moi = 0
    for omkey in pulses.keys():
        qs = pulses[omkey]
        qdom = sum(i.charge for i in qs)
	d = (cogPos - geometry[omkey].position).magnitude**2
	moi = moi+qdom*d

    qoth = qtot-sum(st_info['q'])
 #   print("AAA", moi,qtot,sum(st_info['q']),qoth)
=======
    
>>>>>>> master
    # build image
    # channel 0 is max charge string,
    # channel 1 is max charge neighbor,
    # channel 2 is second highest charge neighbor, etc

<<<<<<< HEAD
    # wfms = []
    # min_time = None
    # for img_ch, (q, stnum, dist) in enumerate(st_info):
    #     for omkey in wf_map.keys():
    # 	    if (omkey.string == stnum):
    # 		for wf in wf_map.get(omkey, []):
    #                 if wf.status == 0 and wf.source_index == 0:   
    # 			if min_time is None or wf.time < min_time:
    #                         min_time = wf.time

    # 			wfms.append({
    # 				'wfm': wf.waveform,
    #                             'time': wf.time,  
    #                             'width': wf.bin_width,
    #                             'dom_idx': omkey.om - 1,
    #                             'img_ch': img_ch
    #                             })
    # 		    else:
    # 		        print(img_ch, wf.status,wf.source_index)   

#    generator = from_simprod(dataset)
#    flux = GaisserH4a()
#    en = weight[0]['PrimaryEnergy']
#    pt = weight[0]['PrimaryType']
#    print(en,pt)
#    we = flux(en, pt)/generator(en, pt)
#    print("AAAAA",we)

    wfms = []
    wf_times = []
    for img_ch, (q, stnum, dist) in enumerate(st_info):
        for omkey in wf_map.keys():
	    if (omkey.string == stnum):
		for wf in wf_map.get(omkey, []):
                    if wf.status == 0 and wf.source_index == 0:   
                        wf_times.append(wf.time)
=======
    wfms = []
    min_time = None
    for img_ch, (q, stnum, dist) in enumerate(st_info):
#        print(img_ch, (q, stnum, dist))
        for omkey in wf_map.keys():
	    if (omkey.string == stnum):
		for wf in wf_map.get(omkey, []):
                    if wf.status == 0 and wf.source_index == 0:
			if min_time is None or wf.time < min_time:
                            min_time = wf.time

>>>>>>> master
			wfms.append({
				'wfm': wf.waveform,
                                'time': wf.time,  
                                'width': wf.bin_width,
                                'dom_idx': omkey.om - 1,
                                'img_ch': img_ch
                                })

<<<<<<< HEAD
		    #else:
		        #print(img_ch, wf.status,wf.source_index)   

    im = np.zeros(shape=(N_X_BINS, N_Y_BINS, N_CHANNELS))
    
    #work out min time
    wf_times.sort()
    diff_times = np.diff(wf_times[:5])
    max_diff_pos = np.argmax(diff_times)
    #print("CCC",diff_times[max_diff_pos],diff_times)
    if diff_times[max_diff_pos] > 30:
	min_time = wf_times[max_diff_pos+1]
    else:
        min_time = wf_times[0]

    #print("BBB",min_time)
    #print(wf_times)

    arr0 = []
    if len(wfms) ==0:
	print("Zero WF")

    
    for wfm in wfms:
        wf_shift = 0
        start_ind = min(N_X_BINS, int((wfm['time'] - min_time) / wfm['width']))
        if start_ind <0: 
	    wf_shift = abs(start_ind)
	    start_ind = 0
	end_ind = min((N_X_BINS, start_ind + len(wfm['wfm']) - wf_shift))
	if end_ind <0: 
	    end_ind = 0
	    wfm_vals = []
	#print(start_ind,end_ind,wf_shift)
        else:
	    wfm_vals = wfm['wfm'][wf_shift:end_ind - start_ind + wf_shift]
	#print(len(wfm_vals))
        im[start_ind:end_ind, wfm['dom_idx'], wfm['img_ch']] = wfm_vals
       
	arr0.append([wfm['img_ch'],start_ind,end_ind])
    
    #print(arr0)
    
    im = np.true_divide(im, 10**(-8))
    im = im.astype(np.float32)
    if np.sum(im[:,:,0])==0:
	print("no image 0")
    if np.sum(im[:,:,1])==0:
	print("no image 1")
    if np.sum(im[:,:,2])==0:
	print("no image 2")
    
    ti = [0,0,0,0]
    if frame.Has("tiParams"):
        ti[0] = frame["tiParams"].mineval
        ti[1] = frame["tiParams"].evalratio      
        ti[2] = frame["tiParams"].eval2
        ti[3] = frame["tiParams"].eval3
    
    #Log all the event info
    event = np.zeros(1,dtype = info_dtype)    
    event[["id","image","qtot","qoth","cog","moi","ti","qst","qst_all","map","primary","prim_daughter","primary_child_energy","primary_child_pdg","logan_veto","hese_old","hese","weight"]]=(id[0],im,qtot,qoth,[cogPos.x,cogPos.y,cogPos.z],moi,ti,st_info,st_info_all,emaps,primary[0],prim_daughter[0],energies,pdgs,veto[0],hese[0],hese[1],weight[0])
#    print("AAAA", event['qtot'],event['qoth'],event['cog'],event['moi'],"aa",event["ti"],event['map'],event['hese'],event['logan_veto'],event['hese_old'])
=======
 

    im = np.zeros(shape=(N_X_BINS, N_Y_BINS, N_CHANNELS))

    for wfm in wfms:
        start_ind = min(N_X_BINS, int((wfm['time'] - min_time) / wfm['width']))
        end_ind = min((N_X_BINS, start_ind + len(wfm['wfm'])))
        wfm_vals = wfm['wfm'][:end_ind - start_ind]
        im[start_ind:end_ind, wfm['dom_idx'], wfm['img_ch']] = wfm_vals
        
    im = np.true_divide(im, 10**(-8))
    im = im.astype(np.float32)


    #Log all the event info
    event = np.zeros(1,dtype = info_dtype)    
    event[["id","image","qtot","qst","primary","prim_daughter","primary_child_energy","primary_child_pdg","logan_veto","hese_old","hese","weight"]]=(id[0],im,qtot,st_info,primary[0],prim_daughter[0],energies,pdgs,veto[0],hese[0],hese[1],weight[0])
    #print("aaa",event['qtot'],event['hese1'],event['hese2'])
>>>>>>> master
    data.append(event)

#@icetray.traysegment
def TestCuts(file_list):
    tray = I3Tray()
    tray.AddModule("I3Reader","reader", FilenameList = file_list)
    tray.AddModule(Check_Data, "check-raw-data", Streams=[icetray.I3Frame.Physics])
    tray.AddModule(Get_Charges, "cuts-and-secelt", Streams=[icetray.I3Frame.Physics])
    tray.AddSegment(QTot.CalQTot, "selfveto-qtot", pulses='SplitInIcePulses')
    tray.AddSegment(Reconstruction.OfflineCascadeReco, "CscdReco", suffix="_DP", Pulses='HLCPulses')
    tray.AddSegment(Reconstruction.MuonReco, "MuonReco", Pulses='HLCPulses')
    tray.AddSegment(Reconstruction.OfflineCascadeReco_noDC, "CscdReco_noDC", suffix="_noDC_DP", Pulses='HLCPulses')
    tray.AddSegment(Reconstruction.MuonReco_noDC, "MuonReco_noDC", Pulses='HLCPulses')
<<<<<<< HEAD
    tray.AddSegment(PolygonContainment.PolygonContainment, 'polyfit', geometry = GEO,RecoVertex='HESE3_VHESelfVetoVertexPos',outputname='_Veto')
    tray.AddModule("I3TensorOfInertia",InputReadout = "SplitInIcePulses")
=======
    tray.AddSegment(PolygonContainment.PolygonContainment, 'polyfit', geometry = GEO,RecoVertex='VHESelfVetoVertexPos',outputname='_Veto')
>>>>>>> master
    tray.AddModule("I3WaveCalibrator", "calibrator")(
        ("Launches", "InIceRawData"),  # EHE burn sample IC86
        ("Waveforms", "CalibratedWaveforms"),
        ("WaveformRange", "CalibratedWaveformRange_DP"),
        ("ATWDSaturationMargin",123), # 1023-900 == 123
        ("FADCSaturationMargin",  0), # i.e. FADCSaturationMargin
        ("Errata", "OfflineInIceCalibrationErrata"), # SAVE THIS IN L2 OUTPUT?
        )
    tray.AddModule("I3WaveformSplitter", "waveformsplit")(
        ("Input","CalibratedWaveforms"),
        ("HLC_ATWD","CalibratedWaveformsHLCATWD"),
        ("HLC_FADC","CalibratedWaveformsHLCFADC"),
        ("SLC","CalibratedWaveformsSLC"),
        ("PickUnsaturatedATWD",True),
        ("Force",True),
        )
    tray.Add(Make_Image, "getwave", Streams=[icetray.I3Frame.Physics])
<<<<<<< HEAD
#    tray.AddModule('I3Writer', 'writer', Filename= outfile+'.i3.bz2', Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics], DropOrphanStreams=[icetray.I3Frame.DAQ])
    tray.AddModule('TrashCan','thecan')
    tray.Execute()
=======
    #tray.AddModule('I3Writer', 'writer', Filename= outfile+'.i3.bz2', Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics], DropOrphanStreams=[icetray.I3Frame.DAQ])
    tray.AddModule('TrashCan','thecan')
    tray.Execute(20)
>>>>>>> master
    tray.Finish()
    return

TestCuts(file_list = file_list)
print "i3 file done"
data = np.array(data)
np.savez_compressed(outfile+"_data"+".npz", data)
<<<<<<< HEAD
#temp = np.array(temp)
#np.savez_compressed(outfile+"_temp"+".npz", temp)

=======
>>>>>>> master
print "finished", data.shape



