#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT /data/user/dpankova/oscNext/oscNext_meta/build/

import icecube
from I3Tray import *
import numpy as np
import glob
import sys
from icecube import icetray, dataclasses, dataio, WaveCalibrator, common_variables
import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i","--infile",
                    dest="infile",
                    type=str,
                    default=[],
		    nargs = "+",
                    help="[I]nfile name")

parser.add_argument("-o","--outfile",
                    dest="outfile",
                    type=str,
                    default="Ot",
                    help="base name for outfile")

parser.add_argument('-p', '--meson_pdg',
                    dest='pdg',
                    type=int,
                    default=15,
                    help='pdg number of nu: nue = 12, numu =14, nutau =16, pdg number of primary meson produced ny nu: e = 11,mu = 13,tau = 15')

parser.add_argument('-e1', '--min_energy',
                    dest='energy_min',
                    type=int,
                    default=500000,
                    help='minimum energy of primary meson in GeV')

parser.add_argument('-e2', '--max_energy',
                    dest='energy_max',
                    type=int,
                    default=1500000,
                    help= 'maximum energy of primary meson in GeV')

parser.add_argument("-sk","--skip_files",
                    dest="skip",
                    type=str,
                    default=["l2_00001014"],
                    nargs="+",
                    help="skip files with that srting in the name")
args = parser.parse_args()

infiles=args.infile
outfile=args.outfile
part_pdg=args.pdg
energy_min=args.energy_min
energy_max=args.energy_max
skip=args.skip
print "skip", skip

c=0.299792458                                                                                                                       
n=1.3195
v=c/n


file_list = []
count  = 0
#data_file = "/data/ana/Cscd/StartingEvents/NuGen_new/NuTau/medium_energy/IC86_flasher_p1=0.3_p2=0.0/l2/1/l2_000011*.i3.zst"
#data_file = "/data/ana/Cscd/StartingEvents/NuGen_new/NuE/medium_energy/IC86_flasher_p1=0.3_p2=0.0/l2/2/l2_000010*.i3.zst"
gfile = '/data/user/dpankova/double_pulse/GeoCalibDetectorStatus_2013.56429_V1_Modified.i3.gz'
geofile = dataio.I3File(gfile)
file_list.append(gfile)

for filename in infiles:
    skip_it = False
    for sk in skip:
        skip_it = sk in filename
    if not skip_it:
	file_list.append(filename)

i_frame = geofile.pop_frame()
g_frame = geofile.pop_frame()
geometry = g_frame["I3Geometry"].omgeo
print file_list

data = []

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
        ("pdg", np.uint32),
        ("energy", np.float32),
        ("position", np.float32,(3)),
        ("direction", np.float32,(2)),
	("time", np.float32),
	("length", np.float32)
    ]
) 
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
kays_dtype = np.dtype(
    [
        ("passed", np.bool_),
	("header", np.bool_),
	("raw_data", np.bool_),
	("weights", np.bool_),
	("mctree", np.bool_),
	("cvstats", np.bool_),
	("null_split", np.bool_),
	("pulses", np.bool_),
           ]
)

info_dtype = np.dtype(
    [
        ("id", id_dtype),
        ("energy", np.float32),
	("pdg", np.float32),
	("weight", weight_dtype),
	("keys", keys_dtype)
    ]
)

def GetData(frame):
    global data 
    
    has_header = frame.Has("I3EventHeader")
    has_rawdata =  False
    if frame.Has("InIceRawData"):
	try:
	    rd = 0
	    rd = frame["InIceRawData"]
	    if len(rd) != 0:
		has_rawdata = True
	    else: 
		has_rawdata = False
	except:
	    has_rawdata = False
    
    has_weights =  frame.Has("I3MCWeightDict")
    has_mctree = frame.Has("I3MCTree")
    has_stats = frame.Has("CVStatistics")
    has_stream = frame["I3EventHeader"].sub_event_stream != "NullSplit"

    has_pulses = False
    if frame.Has("SplitInIcePulses"):
        try:
            pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulses')                                         
	    if not (len(pulses) == 0):
                has_pulses = True
        except:
            has_pulses = False
    passed = has_header and has_weights and has_rawdata and has_mctree and has_stats and has_stream and has_pulses

    id = np.zeros(1,dtype = id_dtype)
    keys = np.zeros(1,dtype = keys_dtype)
    weight = np.zeros(1,dtype = weight_dtype)
   
    if has_header:
	H = frame["I3EventHeader"]
	id[["run_id","sub_run_id","event_id","sub_event_id"]] = (H.run_id,H.sub_run_id,H.event_id,H.sub_event_id)
    
    if has_mctreee:	
	mctree = frame["I3MCTree"]                                                     
	nu = dataclasses.get_most_energetic_neutrino(mctree)
	nu_chldn = mctree.children(nu.id)
	meson = nu_chldn[0]

    #MAKE Type CUT
	if (meson.pdg_encoding != abs(part_pdg)):
	    return False
    
    #get weights
    w = dict(frame["I3MCWeightDict"])
    
    #assign stuctured types
    event = np.zeros(1,dtype = info_dtype)

    
    
    keys[["passed","header","raw_data","weights","mctree","cvstats","null_split","pulses"]] = (has_header,has_weights,has_rawdata,has_mctree,has_stats,has_stream,has_pulses)
    
    weight[list(w.keys())] = tuple(w.values())
    
    event[["id","image","neutrino","meson","q_tot","cog","q_st","st_pos","st_num","distance"]]=(id[0],im,parts[0],parts[1],qtot,[cog.x,cog.y,cog.z], max_q, [pos_st.x,pos_st.y,pos_st.z],max_st,dist )
   
    
    data.append(event)

#@icetray.traysegment

def TestCuts(file_list):
    tray = I3Tray()
    tray.AddModule("I3Reader","reader", FilenameList = file_list)
    tray.AddModule(CheckData, "check-raw-data", Streams=[icetray.I3Frame.Physics])
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
    #tray.AddModule(CheckData, "check-data", Streams = [icetray.I3Frame.Physics])
    tray.Add(GetWaveform, "getwave", Streams=[icetray.I3Frame.Physics])
 #   tray.AddModule('I3Writer', 'writer', Filename= outfile+'.i3.bz2', Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics], DropOrphanStreams=[icetray.I3Frame.DAQ])
    tray.AddModule('TrashCan','thecan')
    tray.Execute()
    tray.Finish()
    return

TestCuts(file_list = file_list)
print "i3 file done"
data = np.array(data)

mm_array = np.lib.format.open_memmap(outfile+"_data"+".npy", dtype=info_dtype, mode="w+", shape=(data.shape[0],))
np.save(outfile+"_data"+".npy",data)
print "finished", data.shape

# def TestCuts(file_list):
#     tray = I3Tray()
#     tray.AddModule("I3Reader","reader", FilenameList = file_list)
#     tray.AddModule(CheckRawData, "check-raw-data")    
#     tray.AddModule(GetMCTreeInfo, "MCTree")
#     #Check if frame contains Raw Data
#     tray.AddModule('I3Writer', 'writer', Filename=outfile+'.i3.bz2', Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics], DropOrphanStreams=[icetray.I3Frame.DAQ])
#     tray.AddModule('TrashCan','thecan')
#     tray.Execute()
#     tray.Finish()


#     return

#TestCuts(file_list = file_list)


