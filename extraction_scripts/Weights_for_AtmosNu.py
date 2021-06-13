import icecube
from I3Tray import *
from icecube import icetray, dataclasses, dataio
from icecube.gulliver import I3LogLikelihoodFitParams
from icecube.weighting.fluxes import GaisserH4a
from icecube.weighting.weighting import from_simprod
from icecube.icetray import I3Units
icetray.set_log_level(icetray.I3LogLevel.LOG_FATAL)
#vars
import nuflux
import numpy as np
import glob
import sys
import argparse
import re

parser = argparse.ArgumentParser()

parser.add_argument("-i","--inputdir",
                    dest="indir",
                    type=str,
                    default="IN",
                    help="base name for input directory")


parser.add_argument("-o","--outfile",
                    dest="outfile",
                    type=str,
                    default="Out",
                    help="base name for outfile")


args = parser.parse_args()
outfile=args.outfile
indir=args.indir

files = []
f = glob.glob(indir+'*.npz')
print(len(f))
nfiles = 1000

id_dtype = np.dtype(
    [
        ("run_id", np.uint32),
        ("sub_run_id", np.uint32),
        ("event_id", np.uint32),
        ("sub_event_id", np.uint32),
    ]
)    
                        
event_dtype = np.dtype(
    [
        ('id', id_dtype),
        ("energy", np.float32),
        ("qtot", np.float32),
        ("ptype", np.uint32),
#        ("weight_c", np.float32),
        ("weight_p", np.float32),
    ]
)

#flux_conv=nuflux.makeFlux('honda2006').getFlux
flux_prompt = nuflux.makeFlux('BERSS_H3p_central').getFlux
#flux_conv=nuflux.makeFlux('H3a_SIBYLL23C').getFlux
#flux_prompt = nuflux.makeFlux('H3a_SIBYLL21').getFlux
weight_name = 'weight'
#weight_name = 'weight_dict'
events = []
count = 0
for file_name in f:
    print(file_name)
    try:
        x = np.load(file_name, mmap_mode="r")['arr_0']
    except Exception:
        print(file_name)
        pass
    
    for e in x:
        event = np.zeros(1,dtype = event_dtype)
        qtot =e['qtot']
        ids = e['id']
        energy = e[weight_name]['PrimaryNeutrinoEnergy']
        ptype = e[weight_name]['PrimaryNeutrinoType'].astype(np.int32)
        cos_theta = np.cos(e[weight_name]['PrimaryNeutrinoZenith'])
        type_weight = e[weight_name]["TypeWeight"]
        oneweight = e[weight_name]['OneWeight']
        nevts = e[weight_name]['NEvents']
#        print(energy, ptype, cos_theta, oneweight, type_weight, nevts)
#        wc = flux_conv(ptype, energy, cos_theta) * oneweight / (type_weight * nevts * nfiles)
#        wp =0
        wp = flux_prompt(ptype, energy, cos_theta) * oneweight / (type_weight * nevts * nfiles)
        event[["id","energy","qtot","ptype","weight_p"]] = (ids,energy,qtot,ptype,wp)
#        event[["id","energy","qtot","ptype","weight_c",'weight_p']] = (ids,energy,qtot,ptype,wc,wp)
        #print(event)
        events.append(event)

events  = np.array(events)
np.savez_compressed(outfile, events)
print("finished",events.shape)
