import numpy as np

N_PRIM_CHILDREN = 3 
STRINGS_TO_SAVE = 10
N_Y_BINS = 60
N_X_BINS = 500
N_CHANNELS = 3
outer_strings = set([1,2,3,4,5,6,7,13,14,21,22,30,31,40,41,50,51,59,60,67,68,72,73,74,75,76,77,78])

id_dtype = np.dtype(
    [
        ("run_id", np.uint32),
        ("sub_run_id", np.uint32),
        ("event_id", np.uint32),
        ("sub_event_id", np.uint32),
    ]
)
preds_dtype = np.dtype(
    [     
        ('n1', np.float32),
        ('n2', np.float32),
#        ('n2_2', np.float32),
#        ('n2_3', np.float32),
        ('n3', np.float32)
    ]
)
st_info_dtype = np.dtype(
    [
        ('q', np.float32),
        ('num', np.uint32),
        ('dist', np.float32)
    ]
)
flux_dtype = np.dtype(
    [     
        ('nom', np.float32),
        ('nu', np.float32),
        ('nl', np.float32),
        ('su', np.float32),
        ('sl', np.float32)
    ]
)
map_dtype = np.dtype(
    [
        ("id", id_dtype),
        ('raw', np.int32),
        ('st_raw', np.int32,(3)),
        ('pulses', np.int32),
        ('st_pulses', np.int32,(3)),
        ('cal', np.int32),
        ('st_cal', np.int32,(3)),
        ('hlc', np.int32),
        ('st_hlc', np.int32,(3)),
        ('slc', np.int32),
        ('st_slc', np.int32,(3))
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

hese_old_dtype = np.dtype(                                             
    [                                                                             
        ("qtot", np.float32),
        ("vheselfveto", np.bool_),
        ("vheselfvetovertexpos", np.float32,(3)),
        ("vheselfvetovertextime", np.float32),
        ("llhratio", np.float32)
    ]
)

hese_dtype = np.dtype(                                             
    [                                                                             
        ("vheselfveto", np.bool_),
        ("vheselfvetovertexpos", np.float32,(3)),
        ("vheselfvetovertextime", np.float32),
    ]
)

CWEIGHT_KEY = "CorsikaWeightMap"

#Corsika 2016
y2016_weight_dtype = np.dtype(
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

#Coriska 2011
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
#Corsika set 10668
d10668_weight_dtype = np.dtype(
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
#corsika year 2012
y2012_weight_dtype = np.dtype(
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

#Corsika set 12379
d12379_weight_dtype = np.dtype(
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

#genie    
genie_weight_dtype = np.dtype(
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
    
muongun_info_dtype = np.dtype(
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
            ("weight_dict", weight_dtype),
        ]
    )
old_info_dtype = np.dtype(
    [
        ("id", id_dtype),
        ("image", np.float32, (N_X_BINS, N_Y_BINS, N_CHANNELS)),
        ("wf_times",np.float32,(N_Y_BINS, N_CHANNELS)),
        ("wf_pos",np.float32,(3, N_Y_BINS, N_CHANNELS)),
        ("wf_width",np.float32),
        ("qtot", np.float32),
        ("cog", np.float32,(3)),
        ("moi", np.float32),
        ("ti", np.float32,(4)),
        ("qst", st_info_dtype, N_CHANNELS),
        ("qst_all", st_info_dtype, STRINGS_TO_SAVE),
        ("map", map_dtype),
        ("primary", particle_dtype),
        ("prim_daughter", particle_dtype),
        ("trck_reco", particle_dtype),
        ("cscd_reco", particle_dtype),
        ("logan_veto", veto_dtype),
        ("hese_old", hese_old_dtype),
        ("hese", hese_old_dtype),
        ("llhcut",np.float32),
        ("weight_val",np.float32)


    ]
)
data_save_dtype = np.dtype(
    [
        ("id", id_dtype),
        ("preds", preds_dtype),
        ("qtot", np.float32),
        ("qst", st_info_dtype, N_CHANNELS),
        ("logan_veto", veto_dtype),
        ("hese", hese_dtype),
    ]
)

save_dtype = np.dtype(
    [
        ("id", id_dtype),
        ("preds", preds_dtype),
        ("qtot", np.float32),
        ("qst", st_info_dtype, N_CHANNELS),
        ("primary", particle_dtype),
        ("prim_daughter", particle_dtype),
        ("logan_veto", veto_dtype),
        ("hese", hese_old_dtype),
        ("llhcut",np.float32),
        ("weight_dict", genie_weight_dtype),
        ("weight_val_0",flux_dtype),
        ("weight_val_1",flux_dtype),
        ("weight_val_2",flux_dtype)
    ]
)

save_atmos_dtype = np.dtype(
    [
        ("id", id_dtype),
        ("preds", preds_dtype),
        ("qtot", np.float32),
        ("qst", st_info_dtype, N_CHANNELS),
        ("primary", particle_dtype),
        ("prim_daughter", particle_dtype),
        ("logan_veto", veto_dtype),
        ("hese", hese_dtype),
        ("weight_c",np.float32),
        ("weight_p",np.float32),
        ]
)

save_atmos_all_dtype = np.dtype(
    [
        ("id", id_dtype),
        ("preds", preds_dtype),
        ("qtot", np.float32),
        ("qst", st_info_dtype, N_CHANNELS),
        ("primary", particle_dtype),
        ("prim_daughter", particle_dtype),
        ("logan_veto", veto_dtype),
        ("hese", hese_old_dtype),
        ("weight_val",np.float32)
        ]
)