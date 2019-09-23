#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.0/icetray-start
#METAPROJECT /data/user/dpankova/oscNext/oscNext_meta/build

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
                    help='pdg number of primary meson produced my nu: e = 11,mu = 13,tau = 15')

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


file_list = []
count  = 0
#data_file = "/data/ana/Cscd/StartingEvents/NuGen_new/NuTau/medium_energy/IC86_flasher_p1=0.3_p2=0.0/l2/1/l2_000011*.i3.zst"
#data_file = "/data/ana/Cscd/StartingEvents/NuGen_new/NuE/medium_energy/IC86_flasher_p1=0.3_p2=0.0/l2/2/l2_000010*.i3.zst"

for filename in infiles:
    skip_it = False
    for sk in skip:
        skip_it = sk in filename
    if not skip_it:
	file_list.append(filename)

print file_list
count = 0



def CheckRawData(frame):
 #   eh = frame["I3EventHeader"]
 #   if (eh.run_id == 33 and eh.event_id ==544) or (eh.run_id == 69 and eh.event_id ==347) or (eh.run_id == 34 and eh.event_id ==497):
 #       return True
 #   else:
 #       return False

    if frame.Has("InIceRawData"):
        return True
    else:
#        print "Didn't find InIceRawData"
        return False

def GetMCTreeInfo(frame):
    #global count
    if not frame.Has("I3MCTree"):
        return False
    mctree = frame["I3MCTree"]
    neutrino = dataclasses.get_most_energetic_neutrino(mctree)
    neutrino_chldn = mctree.children(neutrino.id)
    tau = 0
    #print neutrino_chldn
    for part in neutrino_chldn:
        if abs(part.pdg_encoding) == part_pdg:
            tau = part
    if tau == 0:
        return False
    if (tau.energy>=energy_min) and (tau.energy<=energy_max):    
        print "got one"
        return True
    else:
        return False
   
    

#@icetray.traysegment
def TestCuts(file_list):
    tray = I3Tray()
    tray.AddModule("I3Reader","reader", FilenameList = file_list)
    tray.AddModule(CheckRawData, "check-raw-data")    
    tray.AddModule(GetMCTreeInfo, "MCTree")
    #Check if frame contains Raw Data
    tray.AddModule('I3Writer', 'writer', Filename=outfile+'.i3.bz2', Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics], DropOrphanStreams=[icetray.I3Frame.DAQ])
    tray.AddModule('TrashCan','thecan')
    tray.Execute()
    tray.Finish()


    return

TestCuts(file_list = file_list)

