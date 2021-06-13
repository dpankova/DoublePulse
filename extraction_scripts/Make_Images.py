#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-start
#METAPROJECT combo/stable


import icecube
from I3Tray import *
from icecube import icetray, dataclasses, dataio, WaveCalibrator, wavereform, MuonGun
from icecube.recclasses import I3LineFitParams
from icecube.recclasses import I3CscdLlhFitParams
from icecube.recclasses import I3TensorOfInertiaFitParams
from icecube.gulliver import I3LogLikelihoodFitParams
from icecube.weighting.fluxes import GaisserH4a
from icecube.icetray import I3Units
from icecube.dataclasses import I3Waveform
icetray.set_log_level(icetray.I3LogLevel.LOG_FATAL)
#icetray.set_log_level(icetray.I3LogLevel.LOG_INFO)
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
                    choices=['genie','corsika','muongun','data'],
                    help="corsika, genie, muongun, data")

parser.add_argument("-y","--year",
                    dest="year",
                    type=int,
                    default=2012,
                    choices=[2011,2012,2016],
                    help="production year, only matters for corsika")

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

args = parser.parse_args()
infiles=args.infile
outfile=args.outfile
gfile=args.gcdfile
it=args.it
dataset=args.dataset
data_type=args.data_type
year= args.year

print(infiles)

#PARAMETERS
GEO = 'ic86'
C=0.299792458                                                                  
N=1.3195
V=C/N
QST0_THRES = 2000 # PE, cut on max string charge
QST1_THRES = 10 # PE, cut on max string charge
QST2_THRES = 10 # PE, cut on max string charge
QTOT_THRES = 1000 #PE, cut on total charge
LLH_THRES = -0.1 #LLh diffrence between spe and cascade reco
DIST_ST_CUT = 150**2 #m, look at string within this distance of the max string

#image size
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

#Add input files to file list 
for files in infiles:
    for filename in glob.glob(files):
        if not ('_IT.i3.' in filename) and not ('_EHE.i3.' in filename):
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
        ("vheselfveto", np.bool_),
        ("vheselfvetovertexpos", np.float32,(3)),
        ("vheselfvetovertextime", np.float32),
    ]
)
#Depending on data type set the key names form the frame.

if data_type =='genie':
    WEIGHT_KEY = "I3MCWeightDict"
    PULSES_KEY = 'SplitInIcePulses'
    MCTREE_KEY = 'I3MCTree'
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
elif data_type =='corsika':
    WEIGHT_KEY = "CorsikaWeightMap"
    MCTREE_KEY = 'I3MCTree'
    if year == 2012:
        PULSES_KEY = 'SplitInIcePulses'
        if dataset == 12379:
            weight_dtype = np.dtype(
            [
                ('AreaSum',np.float32),
                ('Atmosphere',np.float32),
                ('BackgroundI3MCPESeriesMapCount',np.float32),
                ('CylinderLength',np.float32),
                ('CylinderRadius',np.float32),
                ('EnergyPrimaryMax',np.float32),
                ('EnergyPrimaryMin',np.float32),
                ('FluxSum',np.float32),
                ('I3MCPESeriesMapCount',np.float32),
                ('Multiplicity',np.float32),
                ('NEvents',np.float32),
                ('OverSampling',np.float32),
                ('ParticleType',np.float32),
                ('Polygonato',np.float32),
                ('PrimaryEnergy',np.float32),
                ('PrimarySpectralIndex',np.float32),
                ('PrimaryType',np.int64),
                ('ThetaMax',np.float32),
                ('ThetaMin',np.float32),
                ('TimeScale',np.float32),
                ('Weight',np.float32),
            ]
            )
        else:
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
                ("PrimaryType",np.int64),
                ("ThetaMax",np.float32),
                ("ThetaMin" ,np.float32),
                ("TimeScale",np.float32),
                ("Weight",np.float32)
            ]                            
            )    

    elif year == 2011:
        PULSES_KEY = 'OfflinePulses'
        if dataset == 10668:
            weight_dtype = np.dtype(
            [
                ('AreaSum',np.float32),
                ('Atmosphere',np.float32),
                ('CylinderLength',np.float32),
                ('CylinderRadius',np.float32),
                ('DiplopiaWeight',np.float32),
                ('EnergyPrimaryMax',np.float32),
                ('EnergyPrimaryMin',np.float32),
                ('FluxSum',np.float32),
                ('Multiplicity',np.float32),
                ('NEvents',np.float32),
                ('OldWeight',np.float32),
                ('Polygonato',np.float32),
                ('PrimaryEnergy',np.float32),
                ('PrimarySpectralIndex',np.float32),
                ('PrimaryType',np.int64),
                ('ThetaMax',np.float32),
                ('ThetaMin',np.float32),
                ('TimeScale',np.float32),
                ('Weight',np.float32)
            ]
            )       
        else:
            weight_dtype = np.dtype(
            [
                ('AreaSum',np.float32),
                ('Atmosphere',np.float32),
                ('CylinderLength',np.float32),
                ('CylinderRadius',np.float32),
                ('DiplopiaWeight',np.float32),
                ('EnergyPrimaryMax',np.float32),
                ('EnergyPrimaryMin',np.float32),
                ('FluxSum',np.float32),
                ('Multiplicity',np.float32),
                ('NEvents',np.float32),
                ('ParticleType',np.int64),
                ('Polygonato',np.float32),
                ('PrimarySpectralIndex',np.float32),
                ('TimeScale',np.float32),
                ('Weight',np.float32)
            ]
            )

    elif year == 2016:
        PULSES_KEY = 'SplitInIcePulses'
        MCTREE_KEY = 'I3MCTree_preMuonProp'
        weight_dtype = np.dtype(
            [
                ('AreaSum',np.float32),
                ('Atmosphere',np.float32),
                ('CylinderLength',np.float32),
                ('CylinderRadius',np.float32),
                ('EnergyPrimaryMax',np.float32),
                ('EnergyPrimaryMin',np.float32),
                ('FluxSum',np.float32),
                ('Multiplicity',np.float32),
                ('NEvents',np.float32),
                ('OverSampling',np.float32),
                ('ParticleType',np.int64),
                ('Polygonato',np.float32),
                ('PrimaryEnergy',np.float32),
                ('PrimarySpectralIndex',np.float32),
                ('PrimaryType',np.float32),
                ('ThetaMax',np.float32),
                ('ThetaMin',np.float32),
                ('TimeScale',np.float32),
                ('Weight',np.float32)
            ]
        )       
elif data_type == 'muongun':
    WEIGHT_KEY = 'None'
    PULSES_KEY = 'SplitInIcePulses'
    MCTREE_KEY = 'I3MCTree_preMuonProp'
    weight_dtype = np.dtype(
        [
            ('ThisisData',np.float32),
        ]
    )  


else: #data
    WEIGHT_KEY = 'None'
    PULSES_KEY = 'SplitInIcePulses'
    MCTREE_KEY = 'None'
    weight_dtype = np.dtype(
        [
            ('ThisisData',np.float32),
        ]
    )  

#Output data format         
if data_type == 'muongun':
    info_dtype = np.dtype(
        [
            ("id", id_dtype),
            ("image", np.float32, (N_X_BINS, N_Y_BINS, N_CHANNELS)),
            ("qtot", np.float32),
            ("qst", st_info_dtype, N_CHANNELS),
            ("primary", particle_dtype),
            ("prim_daughter", particle_dtype),
            ("logan_veto", veto_dtype),                                                  
            ("hese", hese_dtype),
            ("weight_val", np.float32),        
        ]
    )
     
else:
    info_dtype = np.dtype(
        [
            ("id", id_dtype),
            ("image", np.float32, (N_X_BINS, N_Y_BINS, N_CHANNELS)),
            ("qtot", np.float32),
            ("qst", st_info_dtype, N_CHANNELS),
            ("primary", particle_dtype),
            ("prim_daughter", particle_dtype),
            ("trck_reco", particle_dtype),
            ("cscd_reco", particle_dtype),
            ("logan_veto", veto_dtype),                                                  
            ("hese", hese_dtype),
            ("weight_dict", weight_dtype),        
        ]
    )


def effective_area(frame, model, generator):
    mctree = frame["I3MCTree"]
    primary = mctree.primaries[0]
    muon = mctree.get_daughters(primary)[0]
    bundle = MuonGun.BundleConfiguration(
      [MuonGun.BundleEntry(0, muon.energy)])
    area = 1/generator.generated_events(primary, bundle)
    frame["MCMuon"] = muon
    frame["MuonEffectiveArea"] = dataclasses.I3Double(area)
    weighter = MuonGun.WeightCalculator(model, generator)
    weight  = weighter(primary,bundle)
    frame["MuonWeight"] = dataclasses.I3Double(weight)
    return True

def I3MCTpmp_2_I3MCT(frame):
    frame["I3MCTree"]=frame['I3MCTree_preMuonProp']

def harvest_generators(infiles):
    generator = None
    for fname in infiles:
        f = dataio.I3File(fname)
        fr = f.pop_frame(icetray.I3Frame.Stream('S'))
        f.close()
        if fr is not None:
            for k in fr.keys():
                if("I3TriggerHierarchy" in k): continue #hack due to I3TriggerHierchy bug
                v = fr[k]
                if isinstance(v, MuonGun.GenerationProbability):
                    print ("found MG GP")
                    if generator is None:
                        generator = v
                    else:
                        generator += v
    return generator

if data_type == 'muongun':
    muongun_nfiles={21317:9996,21316:9999,21315:15000}
    generator_set1_infile=glob.glob("/data/sim/IceCube/2016/generated/MuonGun/21315/0000000-0000999/*.i3.zst")
    generator_set1=harvest_generators([generator_set1_infile[0]]); infiles_set1 = muongun_nfiles[21315]
    generator_set2_infile=glob.glob("/data/sim/IceCube/2016/generated/MuonGun/21316/0000000-0000999/*.i3.zst")
    generator_set2=harvest_generators([generator_set2_infile[0]]); infiles_set2 = muongun_nfiles[21316]
    generator_set3_infile=glob.glob("/data/sim/IceCube/2016/generated/MuonGun/21317/0000000-0000999/*.i3.zst")
    generator_set3=harvest_generators([generator_set3_infile[0]]); infiles_set3 = muongun_nfiles[21317]
    mg_generator=((infiles_set1*generator_set1) +
               (infiles_set2*generator_set2) +
               (infiles_set3*generator_set3))
    mg_model = MuonGun.load_model('GaisserH4a_atmod12_SIBYLL')
    
#Skip events that don't have the required info
def Check_Data(frame):
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
    if frame.Has(PULSES_KEY):
        try:
            pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, PULSES_KEY)    
            if not len(pulses) == 0:
                has_pulses = True
        except:
            has_pulses = False

    if MCTREE_KEY == 'None':
        has_mctree = True
    else:
        has_mctree = frame.Has(MCTREE_KEY)

    if WEIGHT_KEY == 'None':
        has_weights = True
    else:
        has_weights = frame.Has(WEIGHT_KEY)

    #Images can be made if event has all the keys, passed == True
    passed = has_header and has_weights and has_rawdata and has_mctree and has_pulses 
    if passed:
#        print("PASSES")
        if data_type == 'genie': #Keep only events with right interaction type
            if frame[WEIGHT_KEY]['InteractionType'] != it:
                return False

        return True
    else:
#        print("Not PASSES", has_header,has_weights,has_rawdata,has_mctree,has_pulses)
        return False


def Get_Charges(frame):
    global qtot
    global st_info

    #Calculate charges for cuts
    pulses= dataclasses.I3RecoPulseSeriesMap.from_frame(frame, PULSES_KEY)
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
            qdom = sum([i.charge for i in qs])
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
    if (max_qst < QST0_THRES) or (qtot < QTOT_THRES):
   #     print("FAILED CHARGE CUT ", max_qst, qtot)
        return False
    
    # find neighboring strings and sort by charge
    # include max charge string in this list
    # strings 11 and 19 are neighboring but almost 150 m apart
    
    near_max_strings = []
    for q, st, xy in string_qs:
        dist_st = np.sum((string_qs[0][2]-xy)**2)
        if dist_st < DIST_ST_CUT:
            near_max_strings.append((q, st, dist_st))

    if len(near_max_strings) < N_CHANNELS:
        print('FAILED Not enough hit stirngs in the event')
        return False

    for ch in range(N_CHANNELS): 
        st_info[['q','num','dist']][ch] = (near_max_strings[ch][0],near_max_strings[ch][1],near_max_strings[ch][2])
    
    if (st_info['q'][1] < QST1_THRES) or (st_info['q'][2] < QST2_THRES):
        return False

    return True

def LLH_cut(frame):
    #make llh cut
    has_llhcut = frame.Has('SPEFit32_DPFitParams') and frame.Has('CascadeLlhVertexFit_DPParams')

    if has_llhcut:
        llhcut = frame['SPEFit32_DPFitParams'].rlogl - frame['CascadeLlhVertexFit_DPParams'].ReducedLlh
    else:
        print('FAILED: No recos')
        return False

    #make llh cut
    if llhcut < LLH_THRES:
 #       print("Failed LLH cut = ",llhcut)
        return False
    else:
        print("Passed LLH cut = ",llhcut)
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
    if data_type in ['genie', 'corsika']:
        w = dict(frame[WEIGHT_KEY])
        weight[list(w.keys())] = tuple(w.values())
    
    #Log MCTree info 
    primary = np.zeros(1,dtype = particle_dtype)
    prim_daughter = np.zeros(1,dtype = particle_dtype)
    if not data_type == 'data':
    #find primary particle
        mctree = frame[MCTREE_KEY]
        daughter = None
        if data_type == 'genie':
            prim = dataclasses.get_most_energetic_neutrino(mctree)
            
            max_energy = 0
            for part in mctree.children(prim.id):
                if (part.energy > max_energy) and (abs(part.pdg_encoding) in [11,12,13,14,15,16,17,18]):
                    max_enegy = part.energy
                    daughter = part
        else:
            prim = dataclasses.get_most_energetic_primary(mctree)
            daughter = dataclasses.get_most_energetic_muon(mctree)
           
        
        #if no children, then daughter is a duplicate of primary
        if  daughter is None:
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
            primary[["tree_id","pdg","energy","position","direction","time","length"]] =\
            ([prim.id.majorID, prim.id.minorID], prim.pdg_encoding, prim.energy,\
             [prim.pos.x,prim.pos.y,prim.pos.z],\
             [prim.dir.zenith,prim.dir.azimuth],prim.time, prim.length)
        
            prim_daughter[["tree_id","pdg","energy","position","direction","time","length"]] =\
            ([daughter.id.majorID, daughter.id.minorID], daughter.pdg_encoding,daughter.energy,\
             [daughter.pos.x,daughter.pos.y,daughter.pos.z],\
             [daughter.dir.zenith,daughter.dir.azimuth],daughter.time,daughter.length)
    
    #Log HESE veto perameters 
    hese = np.zeros(1,dtype = hese_dtype)
    hese_vheselfveto = True
    hese_pos =[-9999,-9999,-9999]
    hese_time = -999
    if frame.Has("HESE3_VHESelfVeto"):
        hese_vheselfveto = frame["HESE3_VHESelfVeto"].value
        hese_pos = [frame["HESE3_VHESelfVetoVertexPos"].x,frame["HESE3_VHESelfVetoVertexPos"].y,frame["HESE3_VHESelfVetoVertexPos"].z]
        hese_time = frame["HESE3_VHESelfVetoVertexTime"].value
    hese[["vheselfveto","vheselfvetovertexpos","vheselfvetovertextime"]][0] =\
    (hese_vheselfveto,hese_pos,hese_time) 
   
    #Log logan's veto parameters
    veto = np.zeros(1,dtype = veto_dtype)
    trck_reco = np.zeros(1,dtype = particle_dtype)
    cscd_reco = np.zeros(1,dtype = particle_dtype)
    trck= dataclasses.I3Particle()
    cscs= dataclasses.I3Particle()
 
    veto_cas_rlogl = 999
    veto_spe_rlogl = -999
    veto_cas_rlogl_ndc = 999
    veto_spe_rlogl_ndc = -999
    veto_fh_z = -999
    veto_svv_z = -999
    veto_ldp = -999
    

 
    if frame.Has('SPEFit32_DPFitParams') and frame.Has('CascadeLlhVertexFit_DPParams'):
        veto_cas_rlogl = frame['CascadeLlhVertexFit_DPParams'].ReducedLlh
        veto_spe_rlogl = frame['SPEFit32_DPFitParams'].rlogl
        trck = frame['CascadeLlhVertexFit_DP']
        cscd = frame['SPEFit32_DP']

    if frame.Has('SPEFit32_noDC_DPFitParams') and frame.Has('CascadeLlhVertexFit_noDC_DPParams'):
        veto_cas_rlogl_ndc = frame['CascadeLlhVertexFit_noDC_DPParams'].ReducedLlh
        veto_spe_rlogl_ndc = frame['SPEFit32_noDC_DPFitParams'].rlogl
       
    if frame.Has('HESE3_VHESelfVetoVertexPos'):
        veto_svv_z = frame['HESE3_VHESelfVetoVertexPos'].z
       
    if frame.Has("LeastDistanceToPolygon_Veto"):
        veto_ldp = frame["LeastDistanceToPolygon_Veto"].value

    if frame.Has('depthFirstHit'):     
        veto_fh_z = frame['depthFirstHit'].value
    
   
    veto[["SPE_rlogl","Cascade_rlogl","SPE_rlogl_noDC", "Cascade_rlogl_noDC","FirstHitZ","VHESelfVetoVertexPosZ","LeastDistanceToPolygon_Veto"]] =\
    (veto_spe_rlogl,veto_cas_rlogl,veto_spe_rlogl_ndc,veto_cas_rlogl_ndc,veto_fh_z,veto_svv_z,veto_ldp)                     
    trck_reco[["tree_id","pdg","energy","position","direction","time","length"]] =\
    ([trck.id.majorID, trck.id.minorID], trck.pdg_encoding, trck.energy,[trck.pos.x,trck.pos.y,trck.pos.z],\
     [trck.dir.zenith,trck.dir.azimuth], trck.time, trck.length)
    cscd_reco[["tree_id","pdg","energy","position","direction","time","length"]] =\
    ([cscd.id.majorID, cscd.id.minorID], cscd.pdg_encoding, cscd.energy,[cscd.pos.x,cscd.pos.y,cscd.pos.z],\
     [cscd.dir.zenith, cscd.dir.azimuth],cscd.time, cscd.length)
    
    pulses= dataclasses.I3RecoPulseSeriesMap.from_frame(frame, PULSES_KEY)
    wf_map = frame["CalibratedWaveformsHLCATWD"]   
    #make image from raw waveforms 
    wfms = []
    
    for img_ch, (q, stnum, dist) in enumerate(st_info):
        for omkey in wf_map.keys():
            if (omkey.string == stnum):
                for wf in wf_map.get(omkey, []):
                    if wf.status == 0: #and wf.source_index == 0:
                        wfms.append({
                                'wfm': wf.waveform,
                                'time': wf.time,  
                                'width': wf.bin_width,
                                'dom_idx': omkey.om - 1,
                                'img_ch': img_ch,
                                'om_pos': [geometry[omkey].position.x,geometry[omkey].position.y,geometry[omkey].position.z]
                                })


    im = np.zeros(shape=(N_X_BINS, N_Y_BINS, N_CHANNELS))
    wf_times_arr = np.zeros(shape=(N_Y_BINS, N_CHANNELS))
    wf_pos_arr = np.zeros(shape=(3,N_Y_BINS, N_CHANNELS))
    
    if data_type == 'data':
        save = []
        calib = frame['I3Calibration']
        status = frame['I3DetectorStatus']
        width = 3.33334
        for om, ps in pulses.items():

            if not om.string in st_info['num']:
                continue

            if om in wf_map:
                has_good_wf = False
                for wf in wf_map.get(om, []):
                    if wf.status == 0:
                        has_good_wf = True
                if has_good_wf:
                    continue

                    
            chl = np.where(st_info['num'] == om.string)
            p_time = np.min([i.time for i in ps])
            min_time = p_time - width*15
            max_time = p_time + width*112
            times = np.linspace(min_time, max_time, 128)

            cal = calib.dom_cal[om]
            stat = status.dom_status[om]
            wf_vals=wavereform.refold_pulses(ps, I3Waveform.ATWD, 0, cal, stat, times, False)
            
            wfms.append({
                    'wfm': wf_vals,
                    'time': min_time,  
                    'width': width,
                    'dom_idx': om.om - 1,
                    'img_ch': chl[0][0],
                    'om_pos': [geometry[om].position.x,geometry[om].position.y,geometry[om].position.z]
                    })

    if len(wfms) < 2:
        print("FAILED only %d WF" %len(wfms) )
        return False

    #we neeed to prevent early noise hits from shifting the actual
    #interaction from the image time frame
    #first work out when the first waveforn starts
    wf_times = np.array([wf['time'] for wf in wfms])
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
        wf_times_arr[wfm['dom_idx'], wfm['img_ch']] = wfm['time']
        wf_pos_arr[0:3,wfm['dom_idx'], wfm['img_ch']] = wfm['om_pos']
        
        
        if wf_shift > 0:
            print("the images were shifted by {0:.3f}".format(wf_shift))
                
    im = np.true_divide(im, 10**(-8))
    im = im.astype(np.float32)
    
    if np.sum(im[:,:,0])==0:
        print("FAILED no image 0")
        return False
    if np.sum(im[:,:,1])==0:
        print("FAILED no image 1")
        return False
    if np.sum(im[:,:,2])==0:
        print("FAILED no image 2")
        return False
   
    event = np.zeros(1,dtype = info_dtype)    
    #Log all the event info
    if data_type == 'muongun':
        w = frame['MuonWeight'].value
        event[["id","image","qtot","qst","primary","prim_daughter","logan_veto","hese","weight_val"]]=\
           (id[0], im, qtot, st_info, primary[0], prim_daughter[0], veto[0],hese[0], w)
    else:
        event[["id","image","qtot","qst","primary","prim_daughter","trck_reco","cscd_reco","logan_veto","hese","weight_dict"]]=\
           (id[0], im, qtot, st_info, primary[0], prim_daughter[0],trck_reco[0],cscd_reco[0],veto[0],hese[0], weight[0])
#    print(event['qtot'],event['qst'],event['logan_veto'])
#    print(event['trck_reco'])
#    print(event['cscd_reco'])
    data.append(event)
    
#@icetray.traysegment
def TestCuts(file_list):
    tray = I3Tray()
    tray.AddModule("I3Reader","reader", FilenameList = file_list)
    tray.AddModule(Check_Data, "check-raw-data", Streams=[icetray.I3Frame.Physics])
    tray.AddSegment(QTot.CalQTot, "selfveto-qtot", pulses= PULSES_KEY)
    tray.AddModule(Get_Charges, "cuts-and-secelt", Streams=[icetray.I3Frame.Physics])
    tray.AddSegment(Reconstruction.OfflineCascadeReco, "CscdReco", suffix="_DP", Pulses='HLCPulses')
    tray.AddSegment(Reconstruction.MuonReco, "MuonReco", Pulses='HLCPulses')
    tray.AddModule(LLH_cut, "llhcut", Streams=[icetray.I3Frame.Physics])
    tray.AddSegment(Reconstruction.OfflineCascadeReco_noDC, "CscdReco_noDC", suffix="_noDC_DP", Pulses='HLCPulses')
    tray.AddSegment(Reconstruction.MuonReco_noDC, "MuonReco_noDC", Pulses='HLCPulses')
    tray.AddSegment(PolygonContainment.PolygonContainment, 'polyfit', geometry = GEO,RecoVertex='HESE3_VHESelfVetoVertexPos',outputname='_Veto')
   # tray.AddModule("I3TensorOfInertia",InputReadout = pulse_series)
    tray.AddModule("I3WaveCalibrator", "calibrator")(        
        ("Launches", "InIceRawData"),  # EHE burn sample IC86
        ("Waveforms", "CalibratedWaveforms"),
        ("WaveformRange", "CalibratedWaveformRange_DP"),
        ("ATWDSaturationMargin",123), # 1023-900 == 123
        ("FADCSaturationMargin",  0), # i.e. FADCSaturationMargin
        ("Errata", "BorkedOMs"), # SAVE THIS IN L2 OUTPUT?
        )
    tray.AddModule("I3WaveformSplitter", "waveformsplit")(
        ("Input","CalibratedWaveforms"),
        ("HLC_ATWD","CalibratedWaveformsHLCATWD"),
        ("HLC_FADC","CalibratedWaveformsHLCFADC"),
        ("SLC","CalibratedWaveformsSLC"),
        ("PickUnsaturatedATWD",True),
        ("Force",True),
        )
    #uncomment Next two for muongun
    #tray.Add(I3MCTpmp_2_I3MCT,Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics])
    #tray.Add(effective_area,model=mg_model,generator=mg_generator,Streams=[icetray.I3Frame.Physics])
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
    llhcut = 0

    sp = re.split('\.|/',file)
    outfilename = outfile+"_"+sp[-4]+"_"+sp[-3]+"_data"+".npz"
    txtfilename = outfile+"_"+sp[-4]+"_"+sp[-3]+"_text"+".txt"
    
    with open(txtfilename, 'w+') as f:
        f.write(file+"\n")
        
    TestCuts(file_list = [gfile,file])
    print(file, " i3 file done")
    
    data = np.array(data)
    with open(txtfilename, 'a') as f:
        f.write(str(data.shape[0])+"\n")
    
    if data.shape[0] == 0:
        print("ZERO EVENTS PASSES, NO IMAGES SAVED")
    else:
        np.savez_compressed(outfilename, data)

    print("finished", data.shape)



