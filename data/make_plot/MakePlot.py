#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT /data/user/dpankova/oscNext/oscNext_meta/build/

import icecube
from I3Tray import *
import numpy as np
import glob
import sys
from icecube import icetray, dataclasses, dataio, tableio, common_variables
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


parser.add_argument("-sk","--skip_files",
                    dest="skip",
                    type=str,
                    default=["00001014"],
                    nargs="+",
                    help="skip files with that srting in the name")
args = parser.parse_args()

infiles=args.infile
outfile=args.outfile
skip=args.skip
print "skip", skip

file_list = []
gfile = '/data/user/dpankova/double_pulse/GeoCalibDetectorStatus_2013.56429_V1_Modified.i3.gz'
geofile = dataio.I3File(gfile)
file_list.append(gfile)

for filename in infiles:
    skip_it = False
    for sk in skip:
        skip_it = sk in filename
    if not skip_it:
	file_list.append(filename)

info = []
i_frame = geofile.pop_frame()
g_frame = geofile.pop_frame()
geometry = g_frame["I3Geometry"].omgeo

info = []
print file_list

	    
def CheckData(frame):
    global keys

    
    has_header = frame.Has("I3EventHeader")
    has_rawdata =  frame.Has("InIceRawData")
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
	    
    if has_header and has_rawdata and has_mctree and has_stats and has_stream and has_pulses:
	return True
    else:
 	return False

def GetInfo(frame):
    global geometry
    global info
    
    stats = frame['CVStatistics']
    cog = stats.cog

    mctree = frame["I3MCTree"]
    neutrino = dataclasses.get_most_energetic_neutrino(mctree)
    neutrino_chldn = mctree.children(neutrino.id)
   
    d_parts = []
    for part in neutrino_chldn:
        d_parts.append(part.pdg_encoding)
        
    H = frame["I3EventHeader"]
    pulses= dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulses')
    omkeys = pulses.keys()
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

        sort_st_charge.append((string_q,string_num))
    
    sort_st_charge.sort(key=lambda x: x[0])
    
    max_q_st = sort_st_charge[-1][1]
    max_q = sort_st_charge[-1][0]
    q_tot = sum(i[0] for i in sort_st_charge)

    pos_x = []                                                    
    pos_y = []                     
    pos_z = [] 

    for om, pulseSeries in pulses:
        if om.string == max_q_st:
            pos_x.append(geometry[om].position.x)                        
            pos_y.append(geometry[om].position.y)                             
            pos_z.append(geometry[om].position.z)       
	    
    pos_st  = dataclasses.I3Position(np.mean(pos_x), np.mean(pos_y), np.mean(pos_z))  
    event_info = {}
    event_info['ID']= (int(H.run_id),int(H.sub_run_id),int(H.event_id),int(H.sub_event_id))
    event_info['Energy']= float(neutrino.energy)
    event_info['PDG']= float(d_parts[0])
    event_info['Qtot']= (float(stats.q_tot_pulses),q_tot)
    event_info['Qdom']= float(stats.q_max_doms)
    event_info['Qst']= float(max_q)
    event_info['Distance']= np.sqrt((cog.x-pos_st.x)**2+(cog.y-pos_st.y)**2+(cog.z-pos_st.z)**2)
    event_info['CoG']= (float(cog.x),float(cog.y),float(cog.z))
    event_info['PosSt']= (float(pos_st.x),float(pos_st.y),float(pos_st.z))
    event_info['StQArr']= tuple(sort_st_charge)
    print event_info
    info.append(event_info)
    	    

#@icetray.traysegment

def TestCuts(file_list):
    tray = I3Tray()
    tray.AddModule("I3Reader","reader", FilenameList = file_list)
    tray.AddModule(CheckData, "check-raw-data", Streams = [icetray.I3Frame.Physics])
    tray.AddModule(GetInfo, "info", Streams = [icetray.I3Frame.Physics])
    tray.AddModule('TrashCan','thecan')
    tray.Execute()
    tray.Finish()
    return

TestCuts(file_list = file_list)
print "done"
output_1 = open(outfile+"_data" + '.pkl',"wb")
pickle.dump(info, output_1, -1)
output_1.close()
print "saved info", len(info)



