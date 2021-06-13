import icecube
from I3Tray import *
from icecube import icetray, dataclasses, dataio
from icecube.gulliver import I3LogLikelihoodFitParams
from icecube.weighting.fluxes import GaisserH4a
from icecube.weighting.weighting import from_simprod
from icecube.icetray import I3Units
icetray.set_log_level(icetray.I3LogLevel.LOG_FATAL)
#vars
import numpy as np
import glob
import sys
import argparse
import re

parser = argparse.ArgumentParser()

parser.add_argument("-o","--outfile",
                    dest="outfile",
                    type=str,
                    default="Out",
                    help="base name for outfile")

args = parser.parse_args()
outfile=args.outfile

#dsets = {12040:23346,12161:98026,12268:99985,12332:98370,12379:34516,
#         9036:99977,9255:99973,9622:99978,
#         20787:98557,20789:80272,20848:98933,20852:95332,20849:9948}

#dsets = {12268:99985}
#dsets = {20904:641367}
dsets = {20904:194531}
#dsets = {10670:2265,11058:28239,11057:74206,11362:34684,11499:99964,
#         11637:99535,11808:100000,11865:99860,11905:100000,11926:100000,
#         11937:100000,11943:33263,12040:23346,12161:98026,12268:99985,
#         12332:98370,12379:34516,9036:99977,9255:99973,9622:99978,
#         10281:95830,10282:21975,10309:14357,10475:38096,10651:39746,
#         10668:3955,10784:39001,10899:30750,
#         20849:9948}

files = []
for s in dsets.keys():
    directory = "/data/user/dpankova/double_pulse/images/Corsika_"+str(s)+"_100x/Images_Corsika_20904_0[0,5]*.npz"
    f = glob.glob(directory)
    print(s,len(f))
    files += f
    
print(len(files))

event_dtype = np.dtype(
    [
        ("set", np.uint32),
        ("energy", np.float32),
        ("qtot", np.float32),
        ("ptype", np.uint32),
	("weight", np.float32),
    ]
)

generator = None
for dataset_number_i, n_files_i in dsets.items():
    if generator is None:
        generator = n_files_i * from_simprod(dataset_number_i)
    else:
        print(dataset_number_i)
        generator += n_files_i * from_simprod(dataset_number_i)

def TruePtype(ptype):
    s = str(ptype)[:6]
    if s == '100002':
        ptype = 1000020040
    elif s == '100007':
        ptype = 1000070140
    elif s == '100008':
        ptype  = 1000080160
    elif s == '100013':
        ptype = 1000130270
    elif s == '100026':
        ptype = 1000260560
    return ptype

print(generator([100000],[1000020040]))
flux=GaisserH4a()
events = []
count = 0
nan_count = 0
zero_count = 0
for file_name in files:
    try:
        x = np.load(file_name, mmap_mode="r")['arr_0']
    except Exception:
        print(file_name)
        pass
    
    spl = file_name.split('_')
    for e in x:
        event = np.zeros(1,dtype = event_dtype)
	qtot =e['qtot']
	dset = int(spl[2])
	w = NaN
        if "PrimaryEnergy" in e['weight_dict'].dtype.fields:
	    energy = e['weight_dict']['PrimaryEnergy']
	else:
	    energy = e['primary']['energy']
	
	if "PrimaryType" in e['weight_dict'].dtype.fields:
	    ptype = e['weight_dict']['PrimaryType']
	else:
	    ptype = e['primary']['pdg']


        #ptype = TruePtype(ptype_0[0])
        w = flux(energy, ptype) / generator(energy, ptype)
        if w ==0:
            print('weight= ', w,'set= ',dset,'ptype= ',ptype,'energy= ',energy,'flux= ',flux(energy, ptype),'gennerator= ',generator(energy, ptype))
            zero_count+=1
	elif np.isnan(w):
            print('weight= ', w,'set= ',dset,'ptype= ',ptype,'energy= ',energy,'flux= ',flux(energy, ptype),'gennerator= ',generator(energy, ptype))
            w=0.0
            nan_count+=1
        else:    
            count+=1
        event[["set","energy","qtot","ptype","weight"]] = (dset,energy,qtot,ptype,w)
        events.append(event)

events  = np.array(events)
np.savez_compressed(outfile, events)
print("finished",events.shape, nan_count, zero_count, count)
