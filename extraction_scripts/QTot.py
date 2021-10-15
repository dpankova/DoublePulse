#!/usr/bin/env python
from I3Tray import *
from icecube import icetray, dataio, dataclasses, DomTools
#import sys, numpy

load('VHESelfVeto')

@icetray.traysegment
def CalQTot(tray, name, pulses='', If=True):
	#pulses = 'SplitOfflinePulses'
	#pulses = 'OfflinePulses' #CORSIKA, nue, numu, nutau

	tray.AddModule('HomogenizedQTot', 'qtot_total', Pulses=pulses)
	#tray.AddModule(lambda fr: fr['QTot'].value > 900, 'qtotcut')
	tray.AddModule('I3LCPulseCleaning', 'cleaning', OutputHLC='HLCPulses',
		       OutputSLC='', Input=pulses, If=lambda frame: 'HLCPulses' not in frame)
	tray.AddModule('VHESelfVeto', 'selfveto3', TimeWindow=3000, VertexThreshold=250,
                       DustLayer=-160, DustLayerWidth=60, VetoThreshold=3, Pulses='HLCPulses',
                       OutputBool='HESE3_VHESelfVeto',
                       OutputVertexTime='HESE3_VHESelfVetoVertexTime',
                       OutputVertexPos='HESE3_VHESelfVetoVertexPos')
        tray.AddModule('VHESelfVeto', 'selfveto2',
                       Pulses='HLCPulses',
                       OutputBool='HESE2_VHESelfVeto',
                       OutputVertexTime='HESE2_VHESelfVetoVertexTime',
                       OutputVertexPos='HESE2_VHESelfVetoVertexPos')
	tray.AddModule('HomogenizedQTot', 'qtot_causal', Pulses=pulses,
		       Output='CausalQTot', VertexTime='VHESelfVetoVertexTime')

	return
