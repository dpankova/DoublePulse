#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT /data/user/dpankova/oscNext/oscNext_meta/build/

import icecube
from I3Tray import *
import numpy as np
import glob
import sys
from icecube import icetray, dataclasses, dataio, WaveCalibrator, common_variables
import util.Reconstruction as Reconstruction
import util.PolygonContainment as PolygonContainment 
import util.QTot as QTot 
import pickle
import argparse

from icecube.recclasses import I3DipoleFitParams
from icecube.recclasses import I3LineFitParams
from icecube.recclasses import I3CLastFitParams
from icecube.recclasses import I3CscdLlhFitParams
from icecube.recclasses import I3TensorOfInertiaFitParams
from icecube.recclasses import I3FillRatioInfo
from icecube.recclasses import CramerRaoParams
from icecube.recclasses import I3StartStopParams
from icecube.gulliver import I3LogLikelihoodFitParams

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
data_type=args.data_type
energy_min=args.energy_min
energy_max=args.energy_max
skip=args.skip

print "skip", skip
geo = 'ic86'
c=0.299792458                                                                  
n=1.3195
v=c/n
QST_THRES = 400
QTOT_THRES = 1000
n_prim_children = 3 #number of daughter particles to get from MCTree Primary
#image size
n_y_bins = 60
n_x_bins = 300

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
        ("llhratio", np.float32)
    ]
)

if data_type =='genie':
    weight_key = "I3MCWeightDict"
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


info_dtype = np.dtype(                                                                           [
        ("id", id_dtype),                                                                           ("image", np.float32, (300, 60)),
        ("qtot", np.float32),
        ("qst", np.float32),
	("primary", particle_dtype),
        ("prim_daughter", particle_dtype),
        ("primary_child_energy", np.float32,(n_prim_children)),
	("primary_child_pdg", np.float32,(n_prim_children)),
        ("logan_veto", veto_dtype),                                                  
	("hese", hese_dtype),                                                  
	("weight", weight_dtype),

    ]
)

#main data array
data = []
qtot = 0 #total event charge
qst = 0 #max string charge
nst = 0 #max string number

#Remove events that don't have the required info or too little charge
def Check_Data(frame):
    global data 
    global qtot
    global qst
    global nst

    #HEADER and STREAM
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
    has_weights = frame.Has(weight_key)

    #Images can be made if event has all the keys, passed == True
    passed = has_header and has_weights and has_rawdata and has_mctree and has_pulses
    if passed:

	if data_type == 'genie':
	    if frame[weight_key]['InteractionType'] != it:
		return False

	#Calculate charges for cuts
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
        qst, nst = string_qs[0]
        qtot = sum(i[0] for i in string_qs)
	
    
	#MAKE Charge CUT
	if (qst < QST_THRES) or (qtot < QTOT_THRES):
	    return False

	return True
    else:
	return False

def Make_Image(frame):
    global data 
    global qtot
    global qst
    global nst
    global geometry
    
    #Log id info 
    id = np.zeros(1,dtype = id_dtype)
    H = frame["I3EventHeader"]
    id[["run_id","sub_run_id","event_id","sub_event_id"]] = (H.run_id,H.sub_run_id,H.event_id,H.sub_event_id)


    #Log Weight info
    weight = np.zeros(1,dtype = weight_dtype)
    w = dict(frame[weight_key])
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
    
    primary[["tree_id","pdg","energy","position","direction","time","length"]] = ([prim.id.majorID, prim.id.minorID], prim.pdg_encoding, prim.energy,[prim.pos.x,prim.pos.y,prim.pos.z],[prim.dir.zenith,prim.dir.azimuth],prim.time, prim.length)
    prim_daughter[["tree_id","pdg","energy","position","direction","time","length"]] = ([daughter.id.majorID, daughter.id.minorID], daughter.pdg_encoding,daughter.energy,[daughter.pos.x,daughter.pos.y,daughter.pos.z],[daughter.dir.zenith,daughter.dir.azimuth],daughter.time,daughter.length)

 
       
    #Log HESE
    hese = np.zeros(1,dtype = hese_dtype)
    hese_qtot = 0 
    hese_vheselfveto = True
    hese_llhratio = 0

    if frame.Has("HESE_VHESelfVeto") and frame.Has("HESE_CausalQTot") and frame.Has("HESE_llhratio"):
	hese_qtot = frame["HESE_CausalQTot"].value
	hese_vheselfveto = frame["HESE_VHESelfVeto"].value
	hese_llhratio = frame["HESE_llhratio"].value
  
    hese[["qtot","vheselfveto","llhratio"]] = (hese_qtot,hese_vheselfveto,hese_llhratio) 


    #Log logan veto
    veto = np.zeros(1,dtype = veto_dtype)
    veto_cas_rlogl = 999
    veto_spe_rlogl = -999
    veto_cas_rlogl_ndc = 999
    veto_spe_rlogl_ndc = -999
    veto_fh_z = 999
    veto_svv_z = 999
    veto_ldp = -999
    if frame.Has('VHESelfVetoVertexPos') and frame.Has('SPEFit32_DPFitParams') and frame.Has('CascadeLlhVertexFit_DPParams') and frame.Has('SPEFit32_noDC_DPFitParams') and frame.Has('CascadeLlhVertexFit_noDC_DPParams') and frame.Has('depthFirstHit') and frame.Has("LeastDistanceToPolygon_Veto"):
          veto_cas_rlogl = frame['CascadeLlhVertexFit_DPParams'].ReducedLlh
	  veto_spe_rlogl = frame['SPEFit32_DPFitParams'].rlogl
	  veto_cas_rlogl_ndc = frame['CascadeLlhVertexFit_noDC_DPParams'].ReducedLlh
	  veto_spe_rlogl_ndc = frame['SPEFit32_noDC_DPFitParams'].rlogl
	  veto_fh_z = frame['depthFirstHit'].value
	  veto_svv_z = frame['VHESelfVetoVertexPos'].z
	  veto_ldp = frame["LeastDistanceToPolygon_Veto"].value


    veto[["SPE_rlogl","Cascade_rlogl","SPE_rlogl_noDC", "Cascade_rlogl_noDC","FirstHitZ","VHESelfVetoVertexPosZ","LeastDistanceToPolygon_Veto"]] = (veto_spe_rlogl,veto_cas_rlogl,veto_spe_rlogl_ndc,veto_cas_rlogl_ndc,veto_fh_z,veto_svv_z,veto_ldp)                    



    #make image from raw waveforms
    pulses= dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulses')
    wf_map = frame["CalibratedWaveformsHLCATWD"]
    omkeys = wf_map.keys()
    st_keys = []
    for omkey in omkeys:
        if (omkey.string == nst) and omkey in pulses:
	    st_keys.append(omkey)

    wfs_data = []
    wfs_info = []
    pos_x = []
    pos_y = []
    pos_z = []
    for omkey in st_keys:
        pos_x.append(geometry[omkey].position.x)
        pos_y.append(geometry[omkey].position.y)
        pos_z.append(geometry[omkey].position.z)
        for wf in wf_map[omkey]:
            if wf.status == 0 and wf.source_index == 0:
                wf_data = np.array(wf.waveform)
                wf_info = np.array([wf.time, wf.bin_width, omkey.om])
                wfs_data.append(wf_data)
                wfs_info.append(wf_info)

    wfs_data = np.array(wfs_data)
    wfs_info = np.array(wfs_info)
    #pos_st  = dataclasses.I3Position(np.mean(pos_x), np.mean(pos_y), np.mean(pos_z))
    #dist = np.sqrt((cog.x-pos_st.x)**2+(cog.y-pos_st.y)**2+(cog.z-pos_st.z)**2)

    #waveform data
    im = np.zeros(shape = (n_x_bins,n_y_bins))
    t_min = np.min(wfs_info[:,0])
    wfs_info = [[int((i[0]-t_min)/i[1]),int(i[2])-1] for i in wfs_info]

    for i, pos in enumerate(wfs_info):
        if pos[0] < n_x_bins and pos[0]+128 < n_x_bins:
            im[pos[0]:pos[0]+128 ,pos[1]] = wfs_data[i][:]
        elif pos[0] < n_x_bins:
            im[pos[0]:,pos[1]] = wfs_data[i][:n_x_bins-pos[0]]

    im = np.true_divide(im, 10**(-8))
    im = im.astype(np.float32)


    #Log all the event info
    event = np.zeros(1,dtype = info_dtype)    
    event[["id","image","qtot","qst","primary","prim_daughter","primary_child_energy","primary_child_pdg","logan_veto","hese","weight"]]=(id[0],im,qtot,qst,primary[0],prim_daughter[0],energies,pdgs,veto[0],hese[0],weight[0])
    #print("aaa",event)
    data.append(event)

#@icetray.traysegment
def TestCuts(file_list):
    tray = I3Tray()
    tray.AddModule("I3Reader","reader", FilenameList = file_list)
    tray.AddModule(Check_Data, "check-raw-data", Streams=[icetray.I3Frame.Physics])
    tray.AddSegment(QTot.CalQTot, "selfveto-qtot", pulses='SplitInIcePulses')
    tray.AddSegment(Reconstruction.OfflineCascadeReco, "CscdReco", suffix="_DP", Pulses='HLCPulses')
    tray.AddSegment(Reconstruction.MuonReco, "MuonReco", Pulses='HLCPulses')
    tray.AddSegment(Reconstruction.OfflineCascadeReco_noDC, "CscdReco_noDC", suffix="_noDC_DP", Pulses='HLCPulses')
    tray.AddSegment(Reconstruction.MuonReco_noDC, "MuonReco_noDC", Pulses='HLCPulses')
    tray.AddSegment(PolygonContainment.PolygonContainment, 'polyfit', geometry = geo,RecoVertex='VHESelfVetoVertexPos',outputname='_Veto')
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



