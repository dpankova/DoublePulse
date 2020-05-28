#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT /data/user/dpankova/oscNext/oscNext_meta/build/

import icecube
from I3Tray import *
import numpy as np
import glob
import sys
from icecube import icetray, dataclasses, dataio, WaveCalibrator
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
                    default=[],
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

data = []
info = []

print file_list
count = 0



def CheckRawData(frame):
    if frame.Has("InIceRawData"):
        return True
    else:
#        print "Didn't find InIceRawData"
        return False

# def GetMCTreeInfo(frame):
#     #global count
#     if not frame.Has("I3MCTree"):
#         return False
#     mctree = frame["I3MCTree"]
#     neutrino = dataclasses.get_most_energetic_neutrino(mctree)
#     neutrino_chldn = mctree.children(neutrino.id)

#     tau = 0
#     #print neutrino_chldn
#     for part in neutrino_chldn:
#         if abs(part.pdg_encoding) == part_pdg:
#             tau = part

  
#     if tau == 0:
#         return False

#     if (neutrino.energy>=energy_min) and (neutrino.energy<=energy_max):    
#         print "got one"
#         return True
#     else:
#         return False

def GetMCTreeInfo(frame):
    if not frame.Has("I3MCTree"):
        print "No MCTree"
        return False
    else:
        mctree = frame["I3MCTree"]
        neutrino = dataclasses.get_most_energetic_neutrino(mctree)
        neutrino_chldn = mctree.children(neutrino.id)
        tau = 0

        for part in neutrino_chldn:
            if abs(part.pdg_encoding) == part_pdg:
                tau = part

        if tau == 0:
            print "No Tau"
            return False

        tau_chldn =  mctree.children(tau.id)
        max_energy_chld = dataclasses.I3Particle()
        max_energy_chld.energy = 0
        for part in tau_chldn:
            if part.energy > max_energy_chld.energy:
                max_energy_chld = part

        frame["DP_Tau"] = tau
        frame["DP_Tau_End"] = max_energy_chld
        return True
    
def GetWaveform(frame):
    global geometry
    global info
    global data
    H = frame["I3EventHeader"]
    stream = H.sub_event_stream
    if stream == "NullSplit":
#	print("no stream")
        return False

    if not frame.Has("Homogenized_QTot") or not frame.Has("SplitInIcePulses"):
        print "No Pulses or Qtot in the frame"
        return False

    try:
        pulses= dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulses')
    except:
        print "Error getting Pulses"
        return False

    qtot =frame["Homogenized_QTot"].value
    
    if not frame.Has('DP_Tau'):
        print "MCTree wasn't searched"
        return False

    tau = frame["DP_Tau"]
    if (tau.time <=0):
        print "Negative time. Interaction before trigger?"
        return False
	
    
    if not frame.Has("CalibratedWaveformsHLCATWD"):
        print "I3WaveformCalibrator didn't work"
        return False
    
#    print("all cheacks")	
    event_info = {}
    event_info['id']= (int(H.run_id),int(H.sub_run_id),int(H.event_id),int(H.sub_event_id))
    event_info['nu_energy']= float(frame["MCPrimary"].energy)
    event_info['tau_energy']= float(tau.energy)
    event_info['qtotal']= float(qtot)
    event_info['tau_position']=(float(tau.pos.x),float(tau.pos.y),float(tau.pos.z))
    event_info['tau_direction']= (float(tau.dir.zenith),float(tau.dir.azimuth))
    event_info['tau_time']=(float(tau.time),float(frame["DP_Tau_End"].time))
    event_info['tau_length']=float(tau.length)
    event_info['index']=len(data)
    
    

    wf_map = frame["CalibratedWaveformsHLCATWD"]
    omkeys = wf_map.keys()
    strings_set = set()
    string_keys = {}
    for omkey in omkeys:
        if omkey.string not in strings_set:
            strings_set.add(omkey.string)
            string_keys[omkey.string] = [omkey]
        else:
            string_keys[omkey.string].append(omkey)

    sort_st_charge = []

    for string_num in string_keys:
        string_q = 0
        for omkey in string_keys[string_num]:
            if not omkey in pulses:
                continue

            qs = pulses[omkey]
            qdom = sum(i.charge for i in qs)
            string_q = string_q + qdom
        sort_st_charge.append([string_q,string_num])

    sort_st_charge.sort(key=lambda x: x[0])
    max_q_st = sort_st_charge[-1][1]
    max_q = sort_st_charge[-1][0]
    print max_q, qtot
 #   if (max_q < 5000) or (qtot < 10000):
 #       return False

 #   print "AAAAAA", max_q, qtot
 #   print frame["I3MCTree"]
 #   print "BBBBBB"


    wfs_data = []
    wfs_info = []
    doms = []
   
    for omkey in string_keys[max_q_st]:
        if not omkey in pulses:
            continue
      
	qs = pulses[omkey]
        qdom = sum(i.charge for i in qs)
        dpos = geometry[omkey].position

        dom = {}
        dom['key'] = (int(omkey.string),int(omkey.om))
        dom['qdom'] = qdom
        dom['dom_position'] = (float(dpos.x),float(dpos.y),float(dpos.z))

        for wf in wf_map[omkey]:
            if wf.status == 0 and wf.source_index == 0:
                doms.append(dom)
                wf_data = np.array(wf.waveform)
                wf_info = np.array([wf.time, wf.bin_width, omkey.om])
                wfs_data.append(wf_data)
                wfs_info.append(wf_info)


    string = {}
    string["number"] = max_q_st
    string["doms"] = np.array(doms)
    string["charge"] = max_q

    wfs_data = np.array(wfs_data)
    wfs_info = np.array(wfs_info)

    n_y_bins = 60
    n_x_bins = 300
    im = np.zeros(shape = (n_x_bins,n_y_bins))

    t_min = np.min(wfs_info[:,0])
    wfs_info = [[int((i[0]-t_min)/i[1]),int(i[2])-1] for i in wfs_info]

    for i, pos in enumerate(wfs_info):
        if pos[0] < n_x_bins and pos[0]+128 < n_x_bins:
            im[pos[0]:pos[0]+128 ,pos[1]] = wfs_data[i][:]
        elif pos[0] < n_x_bins:
            im[pos[0]:,pos[1]] = wfs_data[i][:n_x_bins-pos[0]] 

    data.append(im)
    event_info['strings'] = string
    info.append(event_info)

#@icetray.traysegment

def TestCuts(file_list):
    tray = I3Tray()
    tray.AddModule("I3Reader","reader", FilenameList = file_list)
    tray.AddModule(GetMCTreeInfo, "MCTree")
    #Check if frame contains Raw Data
    tray.AddModule(CheckRawData, "check-raw-data", Streams = [icetray.I3Frame.Physics, icetray.I3Frame.DAQ])
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
    tray.Add(GetWaveform, "getwave", Streams=[icetray.I3Frame.Physics])
 #   tray.AddModule('I3Writer', 'writer', Filename= outfile+'.i3.bz2', Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics], DropOrphanStreams=[icetray.I3Frame.DAQ])
    tray.AddModule('TrashCan','thecan')
    tray.Execute()
    tray.Finish()
    return

TestCuts(file_list = file_list)
print "done"
output_1 = open(outfile+"_info" + '.pkl',"wb")
pickle.dump(info, output_1, -1)
output_1.close()
print "saved info", len(info)
np.save(outfile+"_data"+".npy",data)
print "finished", len(data)

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


