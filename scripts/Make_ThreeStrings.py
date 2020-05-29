#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT /data/user/dpankova/oscNext/oscNext_meta/build/

import icecube
from I3Tray import *
import numpy as np
import glob
import sys
from icecube import icetray, dataclasses, dataio
from icecube import WaveCalibrator, common_variables
import math
import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--infile",
                    dest="infile",
                    type=str,
                    default=[],
                    nargs="+",
                    help="[I]nfile name")

parser.add_argument("-o", "--outfile",
                    dest="outfile",
                    type=str,
                    default="Ot",
                    help="base name for outfile")

parser.add_argument('-it', '--interaction_type',
                    dest='it',
                    type=int,
                    default=1,
                    help='Interaction types are : CC -1, NC -2 ,GR-3')

parser.add_argument('-e1', '--min_energy',
                    dest='energy_min',
                    type=int,
                    default=500000,
                    help='minimum energy of primary meson in GeV')

parser.add_argument('-e2', '--max_energy',
                    dest='energy_max',
                    type=int,
                    default=1500000,
                    help='maximum energy of primary meson in GeV')

parser.add_argument("-sk", "--skip_files",
                    dest="skip",
                    type=str,
                    default=["l2_00001014"],
                    nargs="+",
                    help="skip files with that srting in the name")

args = parser.parse_args()

infiles = args.infile
print(infiles)

outfile = args.outfile
it = args.it
energy_min = args.energy_min
energy_max = args.energy_max
skip = args.skip
print "skip", skip

n_y_bins = 60
n_x_bins = 500
n_channels = 3

q_st_cut = 2000
q_tot_cut = 2000

c = 0.299792458
n = 1.3195
v = c/n


file_list = []
count = 0
# data_file = "/data/ana/Cscd/StartingEvents/NuGen_new/NuTau/medium_energy/IC86_flasher_p1=0.3_p2=0.0/l2/1/l2_000011*.i3.zst"
# data_file = "/data/ana/Cscd/StartingEvents/NuGen_new/NuE/medium_energy/IC86_flasher_p1=0.3_p2=0.0/l2/2/l2_000010*.i3.zst"
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
        ("pdg", np.uint32),
        ("energy", np.float32),
        ("position", np.float32, (3)),
        ("direction", np.float32, (2)),
        ("time", np.float32),
        ("length", np.float32)
    ]
)
weight_dtype = np.dtype(
    [
        ('PrimaryNeutrinoAzimuth', np.float32),
        ('TotalColumnDepthCGS', np.float32),
        ('MaxAzimuth', np.float32),
        ('SelectionWeight', np.float32),
        ('InIceNeutrinoEnergy', np.float32),
        ('PowerLawIndex', np.float32),
        ('TotalPrimaryWeight', np.float32),
        ('PrimaryNeutrinoZenith', np.float32),
        ('TotalWeight', np.float32),
        ('PropagationWeight', np.float32),
        ('NInIceNus', np.float32),
        ('TrueActiveLengthBefore', np.float32),
        ('TypeWeight', np.float32),
        ('PrimaryNeutrinoType', np.float32),
        ('RangeInMeter', np.float32),
        ('BjorkenY', np.float32),
        ('MinZenith', np.float32),
        ('InIceNeutrinoType', np.float32),
        ('CylinderRadius', np.float32),
        ('BjorkenX', np.float32),
        ('InteractionPositionWeight', np.float32),
        ('RangeInMWE', np.float32),
        ('InteractionColumnDepthCGS', np.float32),
        ('CylinderHeight', np.float32),
        ('MinAzimuth', np.float32),
        ('TotalXsectionCGS', np.float32),
        ('OneWeightPerType', np.float32),
        ('ImpactParam', np.float32),
        ('InteractionType', np.float32),
        ('TrueActiveLengthAfter', np.float32),
        ('MaxZenith', np.float32),
        ('InteractionXsectionCGS', np.float32),
        ('PrimaryNeutrinoEnergy', np.float32),
        ('DirectionWeight', np.float32),
        ('InjectionAreaCGS', np.float32),
        ('MinEnergyLog', np.float32),
        ('SolidAngle', np.float32),
        ('LengthInVolume', np.float32),
        ('NEvents', np.uint32),
        ('OneWeight', np.float32),
        ('MaxEnergyLog', np.float32),
        ('InteractionWeight', np.float32),
        ('EnergyLost', np.float32)
    ]
)

STRINGS_TO_SAVE = 20

Q_st_dist_dtype = np.dtype(
    [
        ('qs', np.float32, (STRINGS_TO_SAVE)),
        ('st_nums', np.float32, (STRINGS_TO_SAVE)),
        ('dists', np.float32, (STRINGS_TO_SAVE)),
    ]
)

info_dtype = np.dtype(
    [
        ("id", id_dtype),
        ("image", np.float32, (n_x_bins, n_y_bins, n_channels)),
        ("q_st_dist", Q_st_dist_dtype),
        ("neutrino", particle_dtype),
        ("daughter", particle_dtype),
        ("energies", np.float32, (10)),
        ("pdgs", np.float32, (10)),
        ("q_tot", np.float32),
        ("cog", np.float32, (3)),
        ("q_st", np.float32),
        ("st_pos", np.float32, (3)),
        ("st_num", np.float32),
        ("distance", np.float32),
        ("weight", weight_dtype)
    ]
)


def CheckData(frame):
    has_header = frame.Has("I3EventHeader")
    has_rawdata = False
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

    has_weights = frame.Has("I3MCWeightDict")

    has_mctree = frame.Has("I3MCTree")
    has_stats = frame.Has("CVStatistics")
    has_stream = frame["I3EventHeader"].sub_event_stream != "NullSplit"

    has_pulses = False
    if frame.Has("SplitInIcePulses"):
        try:
            pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(
                frame, 'SplitInIcePulses')
            if not (len(pulses) == 0):
                has_pulses = True
        except:
            has_pulses = False
    # print(has_header,has_weights,has_rawdata,has_mctree,has_stats,has_stream,has_pulses)
    # print(has_header and has_weights and has_rawdata and has_mctree and has_stats and has_stream and has_pulses)
    if has_header and has_weights and has_rawdata and has_mctree and has_stats and has_stream and has_pulses:
        return True
    else:
        return False


def GetWaveform(frame):
    global geometry
    global info
    global data

    if frame['I3MCWeightDict']['InteractionType'] != it:
        return False

    H = frame["I3EventHeader"]

    stats = frame['CVStatistics']
    cog = stats.cog

    mctree = frame["I3MCTree"]
    nu = dataclasses.get_most_energetic_neutrino(mctree)
    nu_chldn = mctree.children(nu.id)

    energies = []
    daughters = []
    pdgs = []

    for part in nu_chldn:
        energies.append(part.energy)
        daughters.append(part)
        pdgs.append(part.pdg_encoding)

    if len(energies) == 0:
        return False

    z = zip(energies, pdgs, daughters)
    zs = sorted(z, key=lambda x: x[0], reverse=True)
    zs = np.array(zs)

    energies = np.zeros(10)
    pdgs = np.zeros(10)

    if len(z) > 10:
        energies = zs[:, 0][:10]
        pdgs = zs[:, 1][:10]
    else:
        energies[:len(zs)] = zs[:, 0]
    pdgs[:len(zs)] = zs[:, 1]

    daughter = zs[0][2]

    H = frame["I3EventHeader"]

    pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(
        frame, 'SplitInIcePulses')
    wf_map = frame["CalibratedWaveformsHLCATWD"]

    omkeys = wf_map.keys()
    strings_set = set()
    string_keys = {}

    # Make a set of all hit strings and a dictinary of doms on those strings
    for omkey in omkeys:
        if omkey.string not in strings_set:
            strings_set.add(omkey.string)
            string_keys[omkey.string] = [omkey]
        else:
            string_keys[omkey.string].append(omkey)

    # Caculate charge of each string
    string_qs = []
    for string, doms in string_keys.items():
        string_q = 0
        for omkey in doms:
            if omkey in pulses:
                qs = pulses[omkey]
                qdom = sum(i.charge for i in qs)
                string_q = string_q + qdom

        # get associated string x-y position
        st_pos = geometry[doms[0]].position
        st_xy = np.array([st_pos.x, st_pos.y])

        string_qs.append((string_q, string, st_xy))

    # sort strings by charge and find max
    string_qs.sort(key=lambda x: x[0], reverse=True)
    max_q, max_st, max_xy = string_qs[0]

    qtot = sum(i[0] for i in string_qs)

    # MAKE Charge CUT
    if (max_q < q_st_cut) or (qtot < q_tot_cut):
        return False

    # find neighboring strings and sort by charge
    # include max charge string in this list
    # strings 11 and 19 are neighboring but almost 150 m apart
    dist2_cut = 150**2
    near_max_strings = []
    for q, st, xy in string_qs:
        dist2 = np.sum((max_xy-xy)**2)
        if dist2 < dist2_cut:
           near_max_strings.append((q, st, dist2))

    # print('neighbor strings')
    # for string_q in near_max_strings:
    #     print('%d, %.2f, %.2f' % (string_q[1], string_q[0], math.sqrt(string_q[-1])))

    # raw_input('...')

    pos_x = []
    pos_y = []
    pos_z = []

    # get position information
    for omkey in string_keys[max_st]:
        if omkey in pulses:
            pos_x.append(geometry[omkey].position.x)
            pos_y.append(geometry[omkey].position.y)
            pos_z.append(geometry[omkey].position.z)
    pos_st = dataclasses.I3Position(
        np.mean(pos_x), np.mean(pos_y), np.mean(pos_z))
    dist = np.sqrt((cog.x-pos_st.x)**2+(cog.y-pos_st.y)**2+(cog.z-pos_st.z)**2)


    #
    # extract Q_st_dist information
    #

    out_Q_sts = np.full((STRINGS_TO_SAVE), 0, dtype=np.float32)
    out_st_nums = np.full((STRINGS_TO_SAVE), -1, dtype=np.float32)
    out_st_dists = np.full((STRINGS_TO_SAVE), 1e12, dtype=np.float32)
    
    Q_sts, st_nums, st_xys = list(zip(*string_qs)) 

    st_dists = np.sqrt([np.sum((max_xy-xy)**2) for xy in st_xys])

    n_to_copy = len(string_qs)
    if n_to_copy > STRINGS_TO_SAVE:
        n_to_copy = STRINGS_TO_SAVE

    out_Q_sts[:n_to_copy] = Q_sts[:n_to_copy]
    out_st_nums[:n_to_copy] = st_nums[:n_to_copy]
    out_st_dists[:n_to_copy] = st_dists[:n_to_copy]

    out_Q_st_dist = np.empty(1, dtype=Q_st_dist_dtype)
    out_Q_st_dist[['qs', 'st_nums', 'dists']] = (
        out_Q_sts, out_st_nums, out_st_dists)

    # build image
    # channel 0 is max charge string, 
    # channel 1 is max charge neighbor,
    # channel 3 is second highest charge neighbor, etc

    wfms = []
    min_time = None
    for img_ch, (q, stnum, dist) in enumerate(near_max_strings[:n_channels]):
        # print('saving string %d with charge %.2f' % (stnum, q))

        for omkey in string_keys.get(stnum, []):
            for wf in wf_map.get(omkey, []):
                if wf.status == 0 and wf.source_index == 0:
                    if min_time is None or wf.time < min_time:
                        min_time = wf.time

                    wfms.append({'wfm': wf.waveform,
                                 'time': wf.time,
                                 'width': wf.bin_width,
                                 'dom_idx': omkey.om - 1,
                                 'img_ch': img_ch
                                 })

    # raw_input('...')

    im = np.zeros(shape=(n_x_bins, n_y_bins, n_channels))

    for wfm in wfms:
        start_ind = min(n_x_bins, int((wfm['time'] - min_time) / wfm['width']))
        end_ind = min((n_x_bins, start_ind + len(wfm['wfm'])))

        wfm_vals = wfm['wfm'][:end_ind - start_ind]
        
        im[start_ind:end_ind, wfm['dom_idx'], wfm['img_ch']] = wfm_vals

    im = np.true_divide(im, 10**(-8))
    im = im.astype(np.float32)

    # get weights
    w = dict(frame["I3MCWeightDict"])

    # assign stuctured types
    event = np.zeros(1, dtype=info_dtype)

    id = np.zeros(1, dtype=id_dtype)
    prim = np.zeros(1, dtype=particle_dtype)
    meson = np.zeros(1, dtype=particle_dtype)
    weight = np.zeros(1, dtype=weight_dtype)

    id[["run_id", "sub_run_id", "event_id", "sub_event_id"]] = (
        H.run_id, H.sub_run_id, H.event_id, H.sub_event_id)

    prim[["pdg", "energy", "position", "direction", "time", "length"]] = (nu.pdg_encoding, nu.energy, [
                                                                          nu.pos.x, nu.pos.y, nu.pos.z], [nu.dir.zenith, nu.dir.azimuth], nu.time, nu.length)

    meson[["pdg", "energy", "position", "direction", "time", "length"]] = (daughter.pdg_encoding, daughter.energy, [
                                                                           daughter.pos.x, daughter.pos.y, daughter.pos.z], 
                                                                           [daughter.dir.zenith, daughter.dir.azimuth], 
                                                                           daughter.time, daughter.length)

    weight[list(w.keys())] = tuple(w.values())

    event[["id", "image", "q_st_dist", "neutrino", "daughter", "energies", "pdgs", "q_tot", "cog", "q_st", "st_pos", "st_num", "distance", "weight"]] = (
        id[0], im, out_Q_st_dist, prim, meson, energies, pdgs, qtot, [cog.x, cog.y, cog.z], max_q, [pos_st.x, pos_st.y, pos_st.z], max_st, dist, weight[0])

    data.append(event)


def TestCuts(file_list):
    tray = I3Tray()
    tray.AddModule("I3Reader", "reader", FilenameList=file_list)
    tray.AddModule(CheckData, "check-raw-data",
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule("I3WaveCalibrator", "calibrator")(
        ("Launches", "InIceRawData"),  # EHE burn sample IC86
        ("Waveforms", "CalibratedWaveforms"),
        ("WaveformRange", "CalibratedWaveformRange_DP"),
        ("ATWDSaturationMargin", 123),  # 1023-900 == 123
        ("FADCSaturationMargin",  0),  # i.e. FADCSaturationMargin
        ("Errata", "OfflineInIceCalibrationErrata"),  # SAVE THIS IN L2 OUTPUT?
    )
    tray.AddModule("I3WaveformSplitter", "waveformsplit")(
        ("Input", "CalibratedWaveforms"),
        ("HLC_ATWD", "CalibratedWaveformsHLCATWD"),
        ("HLC_FADC", "CalibratedWaveformsHLCFADC"),
        ("SLC", "CalibratedWaveformsSLC"),
        ("PickUnsaturatedATWD", True),
        ("Force", True),
    )
    # tray.AddModule(CheckData, "check-data", Streams = [icetray.I3Frame.Physics])
    tray.Add(GetWaveform, "getwave", Streams=[icetray.I3Frame.Physics])
 #   tray.AddModule('I3Writer', 'writer', Filename= outfile+'.i3.bz2', Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics], DropOrphanStreams=[icetray.I3Frame.DAQ])
    tray.AddModule('TrashCan', 'thecan')
    tray.Execute()
    tray.Finish()
    return


TestCuts(file_list=file_list)
print "i3 file done"
data = np.array(data)
# mm_array = np.lib.format.open_memmap(
#     outfile+"_data"+".npy", dtype=info_dtype, mode="w+", shape=(data.shape[0], 1))

if len(data) > 0:
    # mm_array[:] = data[:]
    np.savez_compressed(outfile+"_data"+".npz", data)
    print "finished", data.shape
else:
    print('Finished. No output.')

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

# TestCuts(file_list = file_list)
