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
		    default =["l2_00001014", "011057.001312","011057.003278","011057.004247","011057.005703","011057.008446", "011057.004247","011057.003278","011057.005703","011057.001312","011057.003278","011057.004247","011057.005703"],
                    nargs="+",
                    help="skip files with that srting in the name")
args = parser.parse_args()

infiles=args.infile
outfile=args.outfile
gfile=args.gcdfile
data_type=args.data_type
energy_min=args.energy_min
energy_max=args.energy_max
skip=args.skip

print "skip", skip

c=0.299792458                                                                  
n=1.3195
v=c/n
n_prim_children = 3 #number of daughter particles to get from MCTree Primary
n_type = 10 #number of particles of particular type to get from MCTree
cpdgs_type = 10 #number of children pgds to save per particle of that type
pdg_to_extract = [13,15,16] #NuE=12,NuMu=14,NuTau=16,E=11,Mu=13,Tau=15

#ADD GCD to the start of the input filelist
file_list = []
geofile = dataio.I3File(gfile)
i_frame = geofile.pop_frame()
g_frame = geofile.pop_frame()
geometry = g_frame["I3Geometry"].omgeo
file_list.append(gfile)

print(infiles)
#Add input files to file list and skip bad files
for files in infiles:
    for filename in glob.glob(files):
        skip_it = False
        for sk in skip:
            skip_it = sk in filename
        if not skip_it:
            file_list.append(filename)

#Final file list
print(file_list)



#Define structured types

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

keys_dtype = np.dtype(
    [
        ("passed", np.bool_),
	("header", np.bool_),
	("raw_data", np.bool_),
	("weights", np.bool_),
	("mctree", np.bool_),
	("pulses", np.bool_),
	("conventional", np.bool_),
	("simtrimmer", np.bool_)
    ]
)

hese_dtype = np.dtype(
    [
	("qtot", np.float32),
	("vheselfveto", np.bool_),
        ("llhratio", np.float32)
    ]
)

tree_dtype = np.dtype(
    [
        ("tree_id", np.uint32, (2)),  
        ("parent_id", np.uint32, (2)),  
        ("pdg", np.int32),
        ("parent_pdg", np.int32),
        ("children_pdgs", np.int32,(cpdgs_type)),
        ("energy", np.float32),      
        ("position", np.float32,(3)),
        ("direction", np.float32,(2)),
        ("time", np.float32),
        ("length", np.float32)
    ]
)

if data_type =='genie':
    weight_key = "I3MCWeightDict"
    q_filter_mask_dtype = np.dtype(
	[
	    ("CascadeFilter_13", np.bool_),
	    ("DeepCoreFilter_13", np.bool_),
	    ("EHEAlertFilterHB_15", np.bool_),
	    ("EHEAlertFilter_15", np.bool_),
	    ("EstresAlertFilter_18", np.bool_),
	    ("FSSCandidate_13", np.bool_),
	    ("FSSFilter_13", np.bool_),
	    ("FilterMinBias_13", np.bool_),
	    ("FixedRateFilter_13", np.bool_),
	    ("GFUFilter_17", np.bool_),
	    ("HESEFilter_15", np.bool_),
	    ("HighQFilter_17", np.bool_),
	    ("I3DAQDecodeException", np.bool_),
	    ("IceTopSTA3_13", np.bool_),
	    ("IceTopSTA5_13", np.bool_),
	    ("IceTop_InFill_STA2_17", np.bool_),
	    ("IceTop_InFill_STA3_13", np.bool_),
	    ("InIceSMT_IceTopCoincidence_13", np.bool_),
	    ("LowUp_13", np.bool_),
	    ("MESEFilter_15", np.bool_),
	    ("MonopoleFilter_16", np.bool_),
	    ("MoonFilter_13", np.bool_),
	    ("MuonFilter_13", np.bool_),
	    ("OnlineL2Filter_17", np.bool_),
	    ("SDST_IceTopSTA3_13", np.bool_),
	    ("SDST_IceTop_InFill_STA3_13", np.bool_),
	    ("SDST_InIceSMT_IceTopCoincidence_13", np.bool_),
	    ("ScintMinBias_16", np.bool_),
	    ("SlopFilter_13", np.bool_),
	    ("SunFilter_13", np.bool_),
	    ("VEF_13", np.bool_)
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
else: #if data_type =='corsika'
    weight_key = "CorsikaWeightMap"
    q_filter_mask_dtype = np.dtype(
	[
	    ("CascadeFilter_12", np.bool_),
	    ("DeepCoreFilter_12", np.bool_),
	    ("DeepCoreFilter_TwoLayerExp_12", np.bool_), 
	    ("EHEFilter_12", np.bool_),
	    ("FSSFilter_12", np.bool_),
	    ("FSSFilter_12_candidate", np.bool_),
	    ("FilterMinBias_12", np.bool_), 
	    ("FixedRateFilter_12", np.bool_), 
	    ("GCFilter_12", np.bool_), 
	    ("ICOnlineL2Filter_12", np.bool_), 
	    ("IceTopSTA3_12", np.bool_), 
	    ("IceTopSTA5_12", np.bool_),
	    ("IceTop_InFill_STA3_12", np.bool_), 
	    ("InIceSMT_IceTopCoincidence_12", np.bool_), 
	    ("LowUp_12", np.bool_), 
	    ("MoonFilter_12", np.bool_), 
	    ("MuonFilter_12", np.bool_), 
	    ("PhysicsMinBiasTrigger_12", np.bool_), 
	    ("SDST_FilterMinBias_12", np.bool_), 
	    ("SDST_IceTopSTA3_12", np.bool_),
	    ("SDST_InIceSMT_IceTopCoincidence_12", np.bool_), 
	    ("SDST_NChannelFilter_12", np.bool_), 
	    ("SlopFilterTrig_12", np.bool_), 
	    ("SunFilter_12", np.bool_), 
	    ("VEFFilter_12", np.bool_)
	]
    )

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
	("primary", particle_dtype),
	("part_type_1", tree_dtype, (n_type)), 
        ("part_type_2", tree_dtype, (n_type)),                                                     
        ("part_type_3", tree_dtype, (n_type)),
        ("qst", np.float32),
	("qtot", np.float32),
	("hese_qtot", np.float32),
	("hese_vheselfveto", np.bool_),
	("hese_llhratio", np.float32),
	("prim_daughter", particle_dtype),
        ("primary_child_energy", np.float32,(n_prim_children)),
	("primary_child_pdg", np.float32,(n_prim_children)),
	("hese", hese_dtype),
	("weight", weight_dtype),
	("qfiltermask", q_filter_mask_dtype),
	("keys", keys_dtype)
    ]
)

#util function to extract particles of certain pdg from mctree
def Extract_MCTree_part(mctree,prim_id,pdg):
    tree = np.zeros(n_type,dtype = tree_dtype)
    count = 0
    for part in mctree:
        if abs(part.pdg_encoding) == pdg:
	    if part.id == prim_id:
		continue
	    if count < n_type:    
		parent = mctree.parent(part.id)
		children = mctree.children(part.id)
		cpdgs = np.zeros(cpdgs_type)
		cpdg = []
		for child in children:
		    cpdg.append(child.pdg_encoding)
		cpdg = np.array(cpdg) 	
	    
		if len(cpdg) > cpdgs_type:
		    cpdgs[:cpdgs_type] = cpdg[:cpdgs_type]
		else:
		    cpdgs[:len(cpdg)] = cpdg
	    
		tree[["tree_id","parent_id","pdg","parent_pdg","children_pdgs","energy","position","direction","time","length"]][count] = ([part.id.majorID, part.id.minorID], [parent.id.majorID, parent.id.minorID], part.pdg_encoding, parent.pdg_encoding, cpdgs, part.energy,[part.pos.x,part.pos.y,part.pos.z],[part.dir.zenith,part.dir.azimuth],part.time, part.length)
		count = count +1
    return tree

#main data array
data = []
#Main data extraction function
def GetData(frame):
    global data 
    
#Look at the presence of the keys in the frame

    #HEADER and STREAM
    has_header = frame.Has("I3EventHeader")
    has_stream = frame["I3EventHeader"].sub_event_stream != "NullSplit"
    if not has_stream:
	return False #Dont use NullSplit frames


    #If event has header, log id info (it should always have a header)
    id = np.zeros(1,dtype = id_dtype)
    if has_header:
	H = frame["I3EventHeader"]
	id[["run_id","sub_run_id","event_id","sub_event_id"]] = (H.run_id,H.sub_run_id,H.event_id,H.sub_event_id)


    #If event has weights, log Weight info
    weight = np.zeros(1,dtype = weight_dtype)
    has_weights = frame.Has(weight_key)
    if has_weights: 
	w = dict(frame[weight_key])
	weight[list(w.keys())] = tuple(w.values())


    #If event has weights, log Filter info	
    qfmask = np.zeros(1,dtype = q_filter_mask_dtype) 
    has_filtermask = frame.Has("QFilterMask")
    if has_filtermask:
	fm = dict(frame["QFilterMask"])
   	qfmask[list(fm.keys())] = tuple(fm.values())


    #Check for vars what decide raw data deletion/keeping
    has_conv = (frame.Has('PassedConventional') and frame['PassedConventional'].value == True) 
    has_simtrim = (frame.Has('SimTrimmer') and frame['SimTrimmer'].value == True)
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


    #Check for pulses and calculate charges for cuts
    has_pulses = False
    qst = 0
    qtot = 0
    if frame.Has("SplitInIcePulses"):
        try:
            pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulses')    
	    if not (len(pulses) == 0):
                has_pulses = True
        except:
            has_pulses = False

    if has_pulses:
        pulses= dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulses')
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
            string_qs.append([string_q,string])
        
	#sort strings by charge and find max
        string_qs.sort(key=lambda x: x[0], reverse = True)
        max_q, max_st = string_qs[0]
        qtot = sum(i[0] for i in string_qs)
	qst = max_q


    #Check for MCTree and extract info
    has_mctree = frame.Has("I3MCTree")
    tree_1 = np.zeros(n_type,dtype = tree_dtype)
    tree_2 = np.zeros(n_type,dtype = tree_dtype)
    tree_3 = np.zeros(n_type,dtype = tree_dtype)
    primary = np.zeros(1,dtype = particle_dtype)
    prim_daughter = np.zeros(1,dtype = particle_dtype)
    energy = 0
    pdg = 0
    if has_mctree:
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
	    has_mctree = False
	    print("MCTree has no primary children")
	    energies = np.zeros(n_prim_children)
	    pdgs = np.zeros(n_prim_children)    

	else:
	    zipped = zip(energies,pdgs,daughters)
	    zipped_sort = sorted(zipped, key = lambda x: x[0], reverse =True)
	    zipped_sort = np.array(zipped_sort)
	    energies = np.zeros(n_prim_children)
	    pdgs = np.zeros(n_prim_children)
	   
	    if len(zipped_sort)>n_prim_children:
		energies = zipped_sort[:,0][:n_prim_children]
		pdgs = zipped_sort[:,1][:n_prim_children]
	    else:
		energies[:len(zipped_sort)] = zipped_sort[:,0]
		pdgs[:len(zipped_sort)] = zipped_sort[:,1]

        daughter = zipped_sort[0][2]
    
	tree_1 = Extract_MCTree_part(mctree,prim.id,pdg_to_extract[0])
	tree_2 = Extract_MCTree_part(mctree,prim.id,pdg_to_extract[1])
	tree_3 = Extract_MCTree_part(mctree,prim.id,pdg_to_extract[2])


	primary[["tree_id","pdg","energy","position","direction","time","length"]] = ([prim.id.majorID, prim.id.minorID], prim.pdg_encoding, prim.energy,[prim.pos.x,prim.pos.y,prim.pos.z],[prim.dir.zenith,prim.dir.azimuth],prim.time, prim.length)
	prim_daughter[["tree_id","pdg","energy","position","direction","time","length"]] = ([daughter.id.majorID, daughter.id.minorID], daughter.pdg_encoding,daughter.energy,[daughter.pos.x,daughter.pos.y,daughter.pos.z],[daughter.dir.zenith,daughter.dir.azimuth],daughter.time,daughter.length)
    #Images can be made if event has all the keys, passed == True
    passed = has_header and has_weights and has_rawdata and has_mctree and has_stream and has_pulses

    #Log key info
    keys = np.zeros(1,dtype = keys_dtype)
    keys[["passed","header","raw_data","weights","mctree","pulses", "conventional","simtrimmer"]] = (passed,has_header,has_rawdata,has_weights,has_mctree,has_pulses,has_conv,has_simtrim)
       
    #Log HESE
    hese = np.zeros(1,dtype = hese_dtype)
    hese_qtot = 0
    hese_vheselfveto = True
    hese_llhratio = 0

    if frame.Has("HESE_VHESelfVeto") and frame.Has("HESE_CausalQTot") and frame.Has("HESE_llratio"):
       hese_qtot = frame["HESE_CausalQTot"].value
       hese_vheselfveto = frame["HESE_VHESelfVeto"].value
       hese_llhratio = frame["HESE_llhratio"].value

    hese[["qtot","vheselfveto","llhratio"]] = (hese_qtot,hese_vheselfveto,hese_llhratio)
    
    
    #assign stuctured types
    event = np.zeros(1,dtype = info_dtype)    
    event[["id", "primary", "part_type_1", "part_type_2", "part_type_3", "qst","qtot", "hese_qtot","hese_vheselfveto","hese_llhratio"'prim_daughterr', "primary_child_energy", "primary_child_pdg", 'hese', "weight","qfiltermask","keys"]]=(id[0], primary[0], tree_1, tree_2, tree_3, qst, qtot, prim_daughter, energies, pdgs, hese[0], weight[0], qfmask[0], keys[0])
    #print("aaa",event['primary_child_pdg'], event['part_type_1'])
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
print "finished", data.shape



