#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT /data/user/dpankova/oscNext/oscNext_meta/build/

import icecube
from I3Tray import *
from icecube import icetray, dataclasses, dataio, WaveCalibrator
from icecube.recclasses import I3DipoleFitParams
from icecube.recclasses import I3LineFitParams
from icecube.recclasses import I3CLastFitParams
from icecube.recclasses import I3CscdLlhFitParams
from icecube.recclasses import I3TensorOfInertiaFitParams
from icecube.recclasses import I3FillRatioInfo
from icecube.recclasses import CramerRaoParams
from icecube.recclasses import I3StartStopParams
from icecube.gulliver import I3LogLikelihoodFitParams
from icecube.weighting.fluxes import GaisserH4a
from icecube.weighting.weighting import from_simprod
from icecube.icetray import I3Units
icetray.set_log_level(icetray.I3LogLevel.LOG_FATAL)
#vars
from icecube.common_variables import hit_statistics
import Reconstruction
import PolygonContainment 
import QTot 


import numpy as np
import glob
import sys
import argparse
import re

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

parser.add_argument("-set","--dataset",
                    dest="dataset",
                    type=int,
		    default=-1,
                    help="corsika dataset number")

parser.add_argument('-it', '--interaction_type',
                    dest='it',
                    type=int,
                    default=1,
                    help='Interaction types are : CC -1, NC -2 ,GR-3')

parser.add_argument('-e1', '--min_energy',
                    dest='energy_min',
                    type=int,
                    default=500000,
                    help= 'minimum energy of primary meson in GeV')

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
dataset=args.dataset
data_type=args.data_type
energy_min=args.energy_min
energy_max=args.energy_max
skip=args.skip
print "skip", skip
print(infiles)

#PARAMETERS
GEO = 'ic86'
C=0.299792458                                                                  
N=1.3195
V=C/N
QST_THRES = 400 # PE, cut on max string charge
QTOT_THRES = 1000 #PE, cut on total charge
LLH_THRES = -0.1 #LLh diffrence between spe and cascade reco
DIST_ST_CUT = 150**2 #m, look at string within this distance of the max string

#image size
STRINGS_TO_SAVE = 10
N_Y_BINS = 60
N_X_BINS = 500
N_CHANNELS = 3 #number of strings to make image from

#for image bug fix, where early niose hits can shift the image frame away
#from the event sometimes <1% cases
#How many simultaneous noise hits can happen too early in the event?
#hardly more than 3
N_NOISE_HITS = 3 
#How big of a shift we need to worry about?
#can't be too small or you would be shifting normal waveforms
#out of the image for no reason
MAX_TIME_SHIFT = 700 #ns or ~half an image 

DEFAULT_INDEX = 2.88
DEFAULT_PHI = 2.1467


#ADD GCD to the start of the input filelist
file_list = []
geofile = dataio.I3File(gfile)
i_frame = geofile.pop_frame()
g_frame = geofile.pop_frame()
geofull = g_frame["I3Geometry"]
geometry = g_frame["I3Geometry"].omgeo
#file_list.append(gfile)

#Add input files to file list and skip bad files
for files in infiles:
    print(files)
    for filename in glob.glob(files):
        print(filename)
        skip_it = False
        for sk in skip:
	    if (sk in filename):
	        skip_it = True
        if not skip_it:
	    file_list.append(filename)

print(file_list) #Final file list


#Define structured types
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
	("cog", np.float32,(3)),
	("moi", np.float32),
	("ti", np.float32,(4)),       
        ("qst", st_info_dtype, N_CHANNELS),
	("qst_all", st_info_dtype, STRINGS_TO_SAVE),
	("map", map_dtype),
	("primary", particle_dtype),
        ("prim_daughter", particle_dtype),
        ("trck_reco", particle_dtype),
	("cscd_reco", particle_dtype),
	("logan_veto", veto_dtype),                                                  
	("hese_old", hese_dtype),  
	("hese", hese_dtype),
        ("llhcut",np.float32),
	("wf_time",np.float32),
	("wf_width",np.float32),
	("weight", weight_dtype),
	("weight_val",np.float32)
	
    ]
)



def Get_rates_genie(weights, n_i3_files, spectral_index=DEFAULT_INDEX, phi_0=DEFAULT_PHI):
    generator = from_simprod(dataset) * n_i3_files
    energy = weights["PrimaryNeutrinoEnergy"] 
    ptype = weights["PrimaryNeutrinoType"]
    cos_theta = numpy.cos(weights["PrimaryNeutrinoZenith"])
    p_int = weights["TotalWeight"]
    genweight = p_int/generator(energy, ptype, cos_theta)
    unit = I3Units.cm2/I3Units.m2
    flux = 1e-18*phi_0*(energy/100e3)**(-spectral_index)
    return 0.5*genweight*(flux/unit)

def Get_rates_corsika(weights, n_i3_files):
    generator = from_simprod(dataset)
    generator *= n_i3_files
    flux = GaisserH4a()
    energy = weights['PrimaryEnergy']
    ptype = weights['PrimaryType']
    return flux(energy, ptype)/generator(energy, ptype)


#Remove events that don't have the required info
def Check_Data(frame):
    global n_events 
    #HEADER and STREAM
    has_header = frame.Has("I3EventHeader") #all event should have a header
    has_stream = frame["I3EventHeader"].sub_event_stream != "NullSplit"
    if not has_stream:
	return False #Dont use NullSplit frames

    has_rawdata =  False #data wich stores raw waveform info
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

    has_pulses = False #pulse series to calculate charge cut
    if frame.Has("SplitInIcePulses"):
        try:
            pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulses')    
	    if not (len(pulses) == 0):
                has_pulses = True
        except:
            has_pulses = False

    has_mctree = frame.Has("I3MCTree")
    has_weights = frame.Has(WEIGHT_KEY)
    #llh for the cut were calculated?
    

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

    #Calculate charges for cuts
    pulses= dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulses')
    omkeys = pulses.keys() #all the keys in pulse series
    strings_set = set() #all strings in the event
    string_keys = {} #doms on each string

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
	print("FAILED CHARGE CUT ",max_qst,qtot)
	return False

    for st in range(STRINGS_TO_SAVE): 
        dist_st = np.sum((string_qs[0][2]-string_qs[st][2])**2)
        st_info_all[['q','num','dist']][st] = (string_qs[st][0],string_qs[st][1],dist_st)
    
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
	
    return True

def LLH_cut(frame):
    global llhcut
    #make llh cut
    has_llhcut = frame.Has('SPEFit32_DPFitParams') and frame.Has('CascadeLlhVertexFit_DPParams')
    llhcut = -999
    if has_llhcut:
	llhcut = frame['SPEFit32_DPFitParams'].rlogl - frame['CascadeLlhVertexFit_DPParams'].ReducedLlh
    #make llh cut
    if llhcut < LLH_THRES:
	print("Failled LLH cut = ",llhcut)
	return False
    else:
	print("Passed LLH cut = ",llhcut)
	return True

def Make_Image(frame):
    global data 
    global geometry
    global st_info
    global llhcut
    
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
    
    #find primary particle
    mctree = frame["I3MCTree"]
    if data_type == 'genie':
	prim = dataclasses.get_most_energetic_neutrino(mctree)
    else:
	prim = dataclasses.get_most_energetic_muon(mctree)
	
    #get list of children and their energies
    energies = []
    daughters = []
    for part in mctree.children(prim.id):
        energies.append(part.energy)
	daughters.append(part)
	
    #if no children, then daughter is a duplicate of primary
    if len(daughters) == 0:
	print("MCTree has no primary children")
	primary[["tree_id","pdg","energy","position","direction","time","length"]] =\
	([prim.id.majorID, prim.id.minorID], prim.pdg_encoding, prim.energy,\
	 [prim.pos.x,prim.pos.y,prim.pos.z],\
	 [prim.dir.zenith,prim.dir.azimuth],prim.time, prim.length)

	prim_daughter[["tree_id","pdg","energy","position","direction","time","length"]] =\
	([prim.id.majorID, prim.id.minorID], prim.pdg_encoding, prim.energy,\
	 [prim.pos.x,prim.pos.y,prim.pos.z],\
	 [prim.dir.zenith,prim.dir.azimuth], prim.time, prim.length)

    #if there are children, daughter is the child with highest energy
    else:
	temp = zip(energies,daughters)
	temp = sorted(temp, key = lambda x: x[0], reverse =True)
	temp = np.array(temp)
	daughter = temp[0][1]
    
	primary[["tree_id","pdg","energy","position","direction","time","length"]] =\
	([prim.id.majorID, prim.id.minorID], prim.pdg_encoding, prim.energy,\
	 [prim.pos.x,prim.pos.y,prim.pos.z],\
	 [prim.dir.zenith,prim.dir.azimuth],prim.time, prim.length)
	
	prim_daughter[["tree_id","pdg","energy","position","direction","time","length"]] =\
	([daughter.id.majorID, daughter.id.minorID], daughter.pdg_encoding,daughter.energy,\
	 [daughter.pos.x,daughter.pos.y,daughter.pos.z],\
	 [daughter.dir.zenith,daughter.dir.azimuth],daughter.time,daughter.length)
    
    #Log HESE
    hese = np.zeros(11,dtype = hese_dtype)
    hese_qtot = -999 
    hese_vheselfveto = True
    hese_llhratio = -999
    hese_pos =[-9999,-9999,-9999]
    hese_time = -999
    if frame.Has("HESE_VHESelfVeto"):
	hese_qtot = frame["HESE_CausalQTot"].value
	hese_vheselfveto = frame["HESE_VHESelfVeto"].value
	hese_llhratio = frame["HESE_llhratio"].value
	hese_pos = [frame["HESE_VHESelfVetoVertexPos"].x,frame["HESE_VHESelfVetoVertexPos"].y,frame["HESE_VHESelfVetoVertexPos"].z]
	hese_time = frame["HESE_VHESelfVetoVertexTime"].value
    hese[["qtot","vheselfveto","llhratio","vheselfvetovertexpos","vheselfvetovertextime"]][0] =\
    (hese_qtot,hese_vheselfveto,hese_llhratio,hese_pos,hese_time) 

    hese_vheselfveto = True
    hese_pos =[-9999,-9999,-9999]
    hese_time = -999
    if frame.Has("HESE3_VHESelfVeto"):
	hese_vheselfveto = frame["HESE3_VHESelfVeto"].value
	hese_pos = [frame["HESE3_VHESelfVetoVertexPos"].x,frame["HESE3_VHESelfVetoVertexPos"].y,frame["HESE3_VHESelfVetoVertexPos"].z]
	hese_time = frame["HESE3_VHESelfVetoVertexTime"].value
    hese[["qtot","vheselfveto","llhratio","vheselfvetovertexpos","vheselfvetovertextime"]][1] =\
    (hese_qtot,hese_vheselfveto,hese_llhratio,hese_pos,hese_time) 


   
    #Log logan veto
    veto = np.zeros(1,dtype = veto_dtype)
    trck_reco = np.zeros(1,dtype = particle_dtype)
    cscd_reco = np.zeros(1,dtype = particle_dtype)
    trck= dataclasses.I3Particle()
    cscs= dataclasses.I3Particle()
    veto_cas_rlogl = -999
    veto_spe_rlogl = 999
    veto_cas_rlogl_ndc = -999
    veto_spe_rlogl_ndc = 999
    veto_fh_z = -999
    veto_svv_z = -999
    veto_ldp = -999
    
    if frame.Has('HESE3_VHESelfVetoVertexPos') and frame.Has('SPEFit32_DPFitParams') and frame.Has('CascadeLlhVertexFit_DPParams')\
    and frame.Has('SPEFit32_noDC_DPFitParams') and frame.Has('CascadeLlhVertexFit_noDC_DPParams') and frame.Has('depthFirstHit')\
    and frame.Has("LeastDistanceToPolygon_Veto"):
	 
        veto_cas_rlogl = frame['CascadeLlhVertexFit_DPParams'].ReducedLlh
	veto_spe_rlogl = frame['SPEFit32_DPFitParams'].rlogl
	veto_cas_rlogl_ndc = frame['CascadeLlhVertexFit_noDC_DPParams'].ReducedLlh
	veto_spe_rlogl_ndc = frame['SPEFit32_noDC_DPFitParams'].rlogl
	veto_fh_z = frame['depthFirstHit'].value
	veto_svv_z = frame['HESE3_VHESelfVetoVertexPos'].z
	veto_ldp = frame["LeastDistanceToPolygon_Veto"].value
	trck = frame['CascadeLlhVertexFit_DP']
	cscd = frame['SPEFit32_DP']
    
    veto[["SPE_rlogl","Cascade_rlogl","SPE_rlogl_noDC", "Cascade_rlogl_noDC","FirstHitZ","VHESelfVetoVertexPosZ","LeastDistanceToPolygon_Veto"]] =\
   (veto_spe_rlogl,veto_cas_rlogl,veto_spe_rlogl_ndc,veto_cas_rlogl_ndc,veto_fh_z,veto_svv_z,veto_ldp)                    

    trck_reco[["tree_id","pdg","energy","position","direction","time","length"]] =\
	([trck.id.majorID, trck.id.minorID], trck.pdg_encoding, trck.energy,\
	 [trck.pos.x,trck.pos.y,trck.pos.z],\
	 [trck.dir.zenith,trck.dir.azimuth], trck.time, trck.length)
    cscd_reco[["tree_id","pdg","energy","position","direction","time","length"]] =\
	([cscd.id.majorID, cscd.id.minorID], cscd.pdg_encoding, cscd.energy,\
	 [cscd.pos.x,cscd.pos.y,cscd.pos.z],\
	 [cscd.dir.zenith, cscd.dir.azimuth],cscd.time, cscd.length)


    #make image from raw waveforms
    pulses= dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulses')
    
    #make image from raw waveforms
    pulses= dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulses')
    wf_map = frame["CalibratedWaveformsHLCATWD"]
    wf_map_slc = frame["CalibratedWaveformsSLC"]
    wf_map_cal = frame["CalibratedWaveforms"]
    rs = frame["InIceRawData"]

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

    moi = -999
    for omkey in pulses.keys():
        qs = pulses[omkey]
        qdom = sum(i.charge for i in qs)
	d = (cogPos - geometry[omkey].position).magnitude**2
	moi = moi+qdom*d
   
    ti = [-999,-999,-999,-999]
    if frame.Has("tiParams"):
        ti[0] = frame["tiParams"].mineval
        ti[1] = frame["tiParams"].evalratio      
        ti[2] = frame["tiParams"].eval2
        ti[3] = frame["tiParams"].eval3
    
    wfms = []
    wf_times = [] #storage for waveforms starting times
    wf_widths = [] #storage for waveform bin widths
    for img_ch, (q, stnum, dist) in enumerate(st_info):
        for omkey in wf_map.keys():
	    if (omkey.string == stnum):
		for wf in wf_map.get(omkey, []):
                    if wf.status == 0 and wf.source_index == 0:
			wf_times.append(wf.time)
			wf_widths.append(wf.bin_width)
			wfms.append({
				'wfm': wf.waveform,
                                'time': wf.time,  
                                'width': wf.bin_width,
                                'dom_idx': omkey.om - 1,
                                'img_ch': img_ch
                                })


    im = np.zeros(shape=(N_X_BINS, N_Y_BINS, N_CHANNELS))
    if len(wfms) ==0:
	print("Zero WF")
	return False

    
    #we neeed to prevent early noise hits from shifting the actual
    #interaction from the image time frame
    #first work out when the first waveforn starts
    wf_times = np.array(wf_times)
    wf_times = wf_times[wf_times.argsort()]
    #find the biggest differnece between starting times
    diff_times = np.diff(wf_times[:N_NOISE_HITS])
    max_diff_pos = np.argmax(diff_times)
    #check if the images needs to be shifted and work out the shift
    if diff_times[max_diff_pos] > MAX_TIME_SHIFT:
    	min_time = wf_times[max_diff_pos+1]
    else:
        min_time = wf_times[0]

    

    #make images
    for wfm in wfms:
        wf_shift = 0
        start_ind = min(N_X_BINS, int((wfm['time'] - min_time) / wfm['width']))
	if start_ind >= 0:
	    end_ind = min(N_X_BINS, start_ind + len(wfm['wfm']))
	    wfm_vals = wfm['wfm'][0:end_ind-start_ind]
	else: 
	    wf_shift = abs(start_ind)
	    start_ind = 0
	    end_ind = min(N_X_BINS, len(wfm['wfm'])-wf_shift)
	    if end_ind <0:
		end_ind = 0
	    wfm_vals = wfm['wfm'][wf_shift:len(wfm['wfm'])]

	im[start_ind:end_ind, wfm['dom_idx'], wfm['img_ch']] = wfm_vals
    	
	
	if wf_shift > 0:
	    print("the images were shifted by {0:.3f}".format(wf_shift))
	        
    im = np.true_divide(im, 10**(-8))
    im = im.astype(np.float32)
    
    if np.sum(im[:,:,0])==0:
	print("no image 0")
	return False
    if np.sum(im[:,:,1])==0:
	print("no image 1")
	return False
    if np.sum(im[:,:,2])==0:
	print("no image 2")
	return False



    #weights
    w = -1
    if not (dataset == -1):
	if data_type == 'genie':
	    w = Get_rates_genie(weight[0], 1, spectral_index=DEFAULT_INDEX, phi_0=DEFAULT_PHI)
	else:
	    w = Get_rates_corsika(weight[0], 1)
    
    #Log all the event info
    event = np.zeros(1,dtype = info_dtype)    
    event[["id","image","qtot","cog","moi","ti","qst","qst_all","map","primary","prim_daughter",\
           "trck_reco","cscd_reco","logan_veto","hese_old","hese","llhcut","wf_time","wf_width","weight","weight_val"]]=\
           (id[0],im,qtot,[cogPos.x,cogPos.y,cogPos.z],moi,ti,st_info,st_info_all,emaps,primary[0],\
            prim_daughter[0],trck_reco[0],cscd_reco[0],veto[0],hese[0],hese[1],llhcut,min_time,np.mean(wf_widths),weight[0],w)
    data.append(event)
    
    # print("TEST")
    # print(event['id']," QTOT= ", event['qtot'])
    # print(np.sum(event['image'][0,:,:,0]),np.sum(event['image'][0,:,:,1]),np.sum(event['image'][0,:,:,2]))  
    # print('PRIMARY')
    # print(event['primary'])
    # print(event['prim_daughter'])
    # print(event['trck_reco'])
    # print(event['cscd_reco'])
    # print("COG= ",event['cog'])
    # print("MOI= ",event['moi']," LLHCUT= ",event['llhcut'], " W= ",event['weight_val'])
    # print("ST_INFO")
    # print(event['qst'])
    # print("ST_INFO_all")
    # print(event['qst_all'])
    # print("TI")
    # print(event['ti'])
    # print("MAP")
    # print(event['map'])
    # print("LOGAN")
    # print(event['logan_veto'])
    # print("HESE")
    # print(event['hese_old'])
    # print(event['hese'])
    # print(event['wf_time'])
    # print(event['wf_width'])    
    # print("WEIGHT")
    # print(event['weight_val'])

#@icetray.traysegment
def TestCuts(file_list):
    tray = I3Tray()
    tray.AddModule("I3Reader","reader", FilenameList = file_list)
    tray.AddModule(Check_Data, "check-raw-data", Streams=[icetray.I3Frame.Physics])
    tray.AddSegment(QTot.CalQTot, "selfveto-qtot", pulses='SplitInIcePulses')
    tray.AddModule(Get_Charges, "cuts-and-secelt", Streams=[icetray.I3Frame.Physics])
    tray.AddSegment(Reconstruction.OfflineCascadeReco, "CscdReco", suffix="_DP", Pulses='HLCPulses')
    tray.AddSegment(Reconstruction.MuonReco, "MuonReco", Pulses='HLCPulses')
    tray.AddModule(LLH_cut, "llhcut", Streams=[icetray.I3Frame.Physics])
    tray.AddSegment(Reconstruction.OfflineCascadeReco_noDC, "CscdReco_noDC", suffix="_noDC_DP", Pulses='HLCPulses')
    tray.AddSegment(Reconstruction.MuonReco_noDC, "MuonReco_noDC", Pulses='HLCPulses')
    tray.AddSegment(PolygonContainment.PolygonContainment, 'polyfit', geometry = GEO,RecoVertex='HESE3_VHESelfVetoVertexPos',outputname='_Veto')
    tray.AddModule("I3TensorOfInertia",InputReadout = "SplitInIcePulses")
    
    tray.AddModule("I3WaveCalibrator", "calibrator")(        ("Launches", "InIceRawData"),  # EHE burn sample IC86
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
#    tray.AddModule('I3Writer', 'writer', Filename= outfile+'.i3.bz2', Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics], DropOrphanStreams=[icetray.I3Frame.DAQ])
    tray.AddModule('TrashCan','thecan')
    tray.Execute()
    tray.Finish()
    return

for file in file_list:
    #main data array
    data = []
    qtot = 0 #total event charge
    st_info = np.zeros(N_CHANNELS,dtype = st_info_dtype) #[Qst, Nst, DistST] for each image
    st_info_all = np.zeros(STRINGS_TO_SAVE,dtype = st_info_dtype) #[Qst, Nst, DistST] 
    llhcut = 0

    sp = re.split('\.|/',file)
    outfilename = outfile+"_"+sp[-4]+"_"+sp[-3]+"_data"+".npz"
    txtfilename = outfile+"_"+sp[-4]+"_"+sp[-3]+"_text"+".txt"

    f = open(txtfilename, "a")
    f.write(file+"\n")
    f.close()

    
    TestCuts(file_list = [gfile,file])
    print(file, " i3 file done")
    
    data = np.array(data)
    f = open(txtfilename, "a")
    f.write(str(data.shape[0])+"\n")
    f.close()

    if data.shape[0] == 0:
	print("ZERO EVENTS PASSES, NO IMAGES SAVED")
    else:
	np.savez_compressed(outfilename, data)

    print("finished", data.shape)



