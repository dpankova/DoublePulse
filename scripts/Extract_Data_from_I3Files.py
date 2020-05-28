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

for files in infiles:
    for filename in glob.glob(files):
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
	("tree_id", np.uint32,(2)),
        ("pdg", np.int32),
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
keys_dtype = np.dtype(
    [
        ("passed", np.bool_),
	("header", np.bool_),
	("raw_data", np.bool_),
	("weights", np.bool_),
	("mctree", np.bool_),
	("cvstats", np.bool_),
	("pulses", np.bool_),
           ]
)


tree_dtype = np.dtype(
    [
        ("tree_id", np.uint32, (2)),  
        ("parent_id", np.uint32, (2)),  
        ("pdg", np.int32),
        ("parent_pdg", np.int32),
        ("children_pdgs", np.int32,(10)),
        ("energy", np.float32),      
        ("position", np.float32,(3)),
        ("direction", np.float32,(2)),
        ("time", np.float32),
        ("length", np.float32)
    ]
)

info_dtype = np.dtype(
    [
        ("id", id_dtype),
	("neutrino", particle_dtype),
	("nutaus", tree_dtype, (10)), 
        ("taus", tree_dtype, (10)),                                                                                                  
        ("energy", np.float32,(3)),
	("pdg", np.float32,(3)),
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

    if not has_stream:
	return False

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
    nu_tree = np.zeros(10,dtype = tree_dtype)
    tau_tree = np.zeros(10,dtype = tree_dtype)
    prim = np.zeros(1,dtype = particle_dtype)
    energy = 0
    pdg = 0


    if has_header:
	H = frame["I3EventHeader"]
	id[["run_id","sub_run_id","event_id","sub_event_id"]] = (H.run_id,H.sub_run_id,H.event_id,H.sub_event_id)
    	
    if has_mctree:	
	mctree = frame["I3MCTree"]                                                     
	nu = dataclasses.get_most_energetic_neutrino(mctree)
	nu_chldn = mctree.children(nu.id)
	pdgs = []
	engs = []
        
	for part in nu_chldn:
            engs.append(part.energy) 
	    pdgs.append(part.pdg_encoding)
            
	if len(engs) == 0:
	    has_mctree = False
	    
	z = zip(engs,pdgs)
	zs = sorted(z, key = lambda x: x[0], reverse =True)
	zs = np.array(zs)
	engs = np.zeros(3)
	pdgs = np.zeros(3)
	   
	#print(len(zs),zs[:,0])
	if len(z)>3:
	    engs = zs[:,0][:3]
	    pdgs = zs[:,1][:3]
	else:
	    engs[:len(zs)] = zs[:,0]
	    pdgs[:len(zs)] = zs[:,1]
	    
	    #pdgs = zs[:,1]
	#print(engs,pdgs)
	nu_count = 0
	tau_count = 0
	for part in mctree:
	    if abs(part.pdg_encoding) == 16:
		if part.id == nu.id:
		    continue
		if nu_count <10:
		    #i mctree.has_parent(part.id)
		    parent = mctree.parent(part.id)
		    nu_children = mctree.children(part.id)
		    cpdgs = np.zeros(10)
		    cpdg = []
		    for c in nu_children:
		        cpdg.append(c.pdg_encoding)
		    cpdg = np.array(cpdg) 	
	    
		    if len(cpdg) > 10:
			cpdgs[:10] = cpdg[:10]
		    else:
			cpdgs[:len(cpdg)] = cpdg
	    
		    nu_tree[["tree_id","parent_id","pdg","parent_pdg","children_pdgs","energy","position","direction","time","length"]][nu_count] = ([part.id.majorID, part.id.minorID], [parent.id.majorID, parent.id.minorID], part.pdg_encoding, parent.pdg_encoding, cpdgs, part.energy,[part.pos.x,part.pos.y,part.pos.z],[part.dir.zenith,part.dir.azimuth],part.time, part.length)
		    nu_count = nu_count +1
	    

	    if abs(part.pdg_encoding) == 15:
		if tau_count <10:
		    parent = mctree.parent(part.id)
		    tau_children = mctree.children(part.id)
		    cpdgs = np.zeros(10)
		    cpdg = []
		    for c in tau_children:
		        cpdg.append(c.pdg_encoding)
		 
		    cpdg = np.array(cpdg) 		    
		    if len(cpdg) > 10:
			cpdgs[:10] = cpdg[:10]
		    else:
			cpdgs[:len(cpdg)] = cpdg

#		    print([part.id.majorID, part.id.minorID],[parent.id.majorID, parent.id.minorID], part.pdg_encoding, parent.pdg_encoding, cpdgs, part.energy,[part.pos.x,part.pos.y,part.pos.z],[part.dir.zenith,part.dir.azimuth], part.time, part.length)
		  	
		    tau_tree[["tree_id","parent_id","pdg","parent_pdg","children_pdgs","energy","position","direction","time","length"]][tau_count] = ([part.id.majorID, part.id.minorID],[parent.id.majorID, parent.id.minorID], part.pdg_encoding, parent.pdg_encoding, cpdgs, part.energy,[part.pos.x,part.pos.y,part.pos.z],[part.dir.zenith,part.dir.azimuth], part.time, part.length)
		    tau_count = tau_count +1

#MAKE Type CUT
#	if (meson.pdg_encoding != abs(part_pdg)):
#	    return False
    
    #get weights
    w = dict(frame["I3MCWeightDict"])
    
    #assign stuctured types
    event = np.zeros(1,dtype = info_dtype)
    
#    print("AAAA", len(nu_tree), nu_tree)
#    print("BBBB", len(tau_tree), tau_tree)
    keys[["passed","header","raw_data","weights","mctree","cvstats","pulses"]] = (passed, has_header,has_weights,has_rawdata,has_mctree,has_stats,has_pulses)
    prim[["tree_id","pdg","energy","position","direction","time","length"]] = ([nu.id.majorID, nu.id.minorID], nu.pdg_encoding, nu.energy,[nu.pos.x,nu.pos.y,nu.pos.z],[nu.dir.zenith,nu.dir.azimuth],nu.time, nu.length)
    weight[list(w.keys())] = tuple(w.values())
    event[["id", "neutrino", "nutaus", "taus", "energy", "pdg", "weight","keys"]]=(id[0], prim, nu_tree, tau_tree, engs, pdgs, weight[0], keys[0])
#    print("CCCC", event)
    
    data.append(event)

#@icetray.traysegment

def TestCuts(file_list):
    tray = I3Tray()
    tray.AddModule("I3Reader","reader", FilenameList = file_list)
    tray.AddModule(GetData, "check-raw-data", Streams=[icetray.I3Frame.Physics])
    #tray.AddModule('I3Writer', 'writer', Filename= outfile+'.i3.bz2', Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics], DropOrphanStreams=[icetray.I3Frame.DAQ])
    tray.AddModule('TrashCan','thecan')
    tray.Execute()
    tray.Finish()
    return

TestCuts(file_list = file_list)
print "i3 file done"
data = np.array(data)
np.savez_compressed(outfile+"_data"+".npz", data)
#mm_array = np.lib.format.open_memmap(outfile+"_data"+".npy", dtype=info_dtype, mode="w+", shape=(data.shape[0],))
#np.save(outfile+"_data"+".npy",data)
print "finished", data.shape



