#!/usr/bin/python
from __future__ import division
import icecube
from I3Tray import *
import numpy as np
import glob
import sys
from icecube import icetray, dataclasses, dataio, phys_services, WaveCalibrator
from argparse import ArgumentParser      
import pickle

name = sys.argv[1]

c=0.299792458
n=1.3195
v=c/n 
   
file_list = []
gfile = '/gpfs/group/dfc13/default/dasha/mlarson/L2/GeoCalibDetectorStatus_2013.56429_V1_Modified.i3.gz'
geofile = dataio.I3File(gfile)
file_list.append(gfile)


#data_file = "/storage/home/d/dup193/work/double_pulse/data/Tau/l2_00000001.i3.zst"
data_file = "/storage/home/d/dup193/work/double_pulse/" + name +".i3.bz2"
#name = "DP_Tau1PeV_Big"
for filename in glob.glob(data_file):
    file_list.append(filename)


i_frame = geofile.pop_frame()
g_frame = geofile.pop_frame()
geometry = g_frame["I3Geometry"].omgeo

data = []
info = []


def CheckRawData(frame):
    if frame.Has("InIceRawData"):
        return True
    else:
        print "Didn't find InIceRawData"
        return False


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
            if abs(part.pdg_encoding) == 11:
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
#    print max_q
    if max_q < 600:
        return False

    im_data = []    
    doms = []

    for omkey in string_keys[max_q_st]:
        if not omkey in pulses:
            continue        
        
        qs = pulses[omkey]
        dpos = geometry[omkey].position
 

        dom = {}
        dom['key'] = (int(omkey.string),int(omkey.om))
        dom['qdom'] = qdom
        dom['dom_position'] = (float(dpos.x),float(dpos.y),float(dpos.z))


     
        for wf in wf_map[omkey]:
            if wf.status == 0 and wf.source_index == 0:
                doms.append(dom)
                wf_vect = list(wf.waveform)
                for i, bin_val in enumerate(wf_vect):
                    sample=[]
                    sample.append(float(dpos.z))
                    sample.append(float((wf.time+wf.bin_width*i)*v))
                    sample.append(float(bin_val*10**12))
                    im_data.append(np.array(sample))


     

    string = {}
    string["number"] = max_q_st
    string["doms"] = np.array(doms)
    string["charge"] = max_q
        
    im_data = np.array(im_data)
    dom_z = np.unique(im_data[:,0])
    n_doms = len(dom_z)
    z_min = dom_z[0]
    z_max = dom_z[-1]
    im_data = im_data[im_data[:, 1].argsort()]
    width = 3.3333
    x_min = im_data[:,1][0]
    x_max = im_data[:,1][-1]
    x_bin = (x_max-x_min)/width+1
        
    im, xed, yed = np.histogram2d(im_data[:,1],im_data[:,0],weights=im_data[:,2],bins=[x_bin,n_doms], range=[[x_min,x_max],[z_min,z_max]])

    data.append(im)




    event_info['strings'] = string
    info.append(event_info)


#    print info
#    print data
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
 #   tray.AddModule('I3Writer', 'writer', Filename= name+'.i3.bz2', Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics], DropOrphanStreams=[icetray.I3Frame.DAQ])
    tray.AddModule('TrashCan','thecan')
    tray.Execute()
    tray.Finish()
    return

TestCuts(file_list = file_list)
print "done"
output_1 = open(name+"_info" + '.pkl',"wb")
pickle.dump(info, output_1, -1)
output_1.close()
print "saved info"
np.save(name+"_data"+".npy",data)
print "finished"
#print data
#print info
