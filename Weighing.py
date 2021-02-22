#Genie and corsika weighting for NuTau double pulse analysis
#Most of it is from icetray
#edited by Daria Pankova
import numexpr

def get_rates_genie(one_weights, nu_E, n_npz_files, i3_per_npz, evts_per_i3file,\
                    spectral_index=DEFAULT_INDEX, phi_0=DEFAULT_PHI):
    ''' returns the per-year weights for the given input parameters '''
    total_events = n_npz_files*i3_per_npz*evts_per_i3file
    secs_per_year = 31536000
    flux_weights = 1e-18*secs_per_year*phi_0*(nu_E/100e3)**(-spectral_index)
    return flux_weights/total_events*one_weights

electronvolt = 1.0e-9;
kiloelectronvolt = 1.e+3*electronvolt;
megaelectronvolt = 1.e+6*electronvolt;
gigaelectronvolt = 1.e+9*electronvolt;
teraelectronvolt = 1.e+12*electronvolt;
petaelectronvolt = 1.e+15*electronvolt;
MeV = megaelectronvolt;
eV = electronvolt;
keV = kiloelectronvolt;
GeV = gigaelectronvolt;
TeV = teraelectronvolt;
PeV = petaelectronvolt;

class baseEnum(int):
    name = None
    values = {}
    def __repr__(self):
        return self.name
    
class metaEnum(type):
    def __new__(cls, classname, bases, classdict):
        newdict = {"values":{}}
        for k in classdict.keys():
            if not (k.startswith('_') or k == 'name' or k == 'values'):
                val = classdict[k]
                member = baseEnum(val)
                member.name = k
                newdict['values'][val] = member
                newdict[k] = member
                    
        # Tell each member about the values in the enum
        for k in newdict['values'].keys():
            newdict['values'][k].values = newdict['values']
        # Return a new class with the "values" attribute filled
        return type.__new__(cls, classname, bases, newdict)
                    
class enum(baseEnum, metaclass=metaEnum):
    """This class mimicks the interface of boost-python-wrapped enums.
       Inherit from this class to construct enumerated types that can
       be passed to the I3Datatype, e.g.:
    class DummyEnummy(tableio.enum):
    Foo = 0
    Bar = 1
    Baz = 2
    
    desc = tableio.I3TableRowDescription()
    desc.add_field('dummy', tableio.I3Datatype(DummyEnummy), '', '') """

class ParticleType(enum):
    PPlus       =   14
    He4Nucleus  =  402
    N14Nucleus  = 1407
    O16Nucleus  = 1608
    Al27Nucleus = 2713
    Fe56Nucleus = 5626
    NuE         =   66
    NuEBar      =   67
    NuMu        =   68
    NuMuBar     =   69
    NuTau       =  133
    NuTauBar    =  134
    
class PDGCode(enum):
    PPlus       =       2212
    He4Nucleus  = 1000020040
    N14Nucleus  = 1000070140
    O16Nucleus  = 1000080160
    Al27Nucleus = 1000130270
    Fe56Nucleus = 1000260560
    NuE         =         12
    NuEBar      =        -12
    NuMu        =         14
    NuMuBar     =        -14
    NuTau       =         16
    NuTauBar    =        -16
PDGCode.from_corsika = classmethod(lambda cls, i: getattr(cls, ParticleType.values[i].name))
ParticleType.from_pdg = classmethod(lambda cls, i: getattr(cls, PDGCode.values[i].name))

def build_lookup(mapping, var='ptype', default='ptype'):
    """
    Build an expression equivalent to a lookup table
    """
    if len(mapping) > 0:
        return 'where(%s==%s, %s, %s)' % (var, mapping[0][0], mapping[0][1],\
                                          build_lookup(mapping[1:], var, default))
    else:
        return str(default)
class CompiledFlux(object):
    """
    An efficient pre-compiled form of a multi-component flux. 
    For single-element evalutions this is ~2 times faster 
    than switching on the primary type with an if statement; for 1e5
    samples it is 2000 times faster than operating on masked slices 
    for each primary type."""
    pdg_to_corsika = numexpr.NumExpr(build_lookup([(int(PDGCode.from_corsika(v)), v) for v in ParticleType.values.keys()]))
    def __init__(self, expr):
        self.expr = numexpr.NumExpr(expr)
        # by default, assume PDG codes
        self._translator = CompiledFlux.pdg_to_corsika
        
    def to_PDG(self):
        """
        Convert to a form that takes PDG codes rather than CORSIKA codes.
        """
        new = copy.copy(self)
        new._translator = CompiledFlux.pdg_to_corsika
        return new
            
    def __call__(self, E, ptype):
        """
        :param E: particle energy in GeV
        :param ptype: particle type code
        :type ptype: int
        """
        if self._translator:
            ptype = self._translator(ptype)
            return self.expr(E, ptype)

    @staticmethod
    def build_lookup(mapping, var='ptype', default=0.):
        """
        Build an expression equivalent to a lookup table
        """
        # force mapping to be a list if it wasn't already
        mapping=list(mapping)
        if len(mapping) > 0:
            return 'where(%s==%s, %s, %s)' % (var, mapping[0][0], mapping[0][1], build_lookup(mapping[1:], var, default))
        else:
            return str(default)
            
class GaisserHillas(CompiledFlux):
    
    ptypes = [getattr(ParticleType, p) for p in ('PPlus', 'He4Nucleus', 'N14Nucleus', 'Al27Nucleus', 'Fe56Nucleus')]
    def get_expression(self, flux, gamma, rigidity):
        z = "where(ptype > 100, ptype%100, 1)"
        return "%(flux)s*E**(-%(gamma)s)*exp(-E/(%(rigidity)s*%(z)s))" % locals()
    def get_flux(self):
        return [[7860., 3550., 2200., 1430., 2120.]]
    def get_gamma(self):
        return [[2.66, 2.58, 2.63, 2.67, 2.63]]
    def get_rigidity(self):
        return [4*PeV]
    def __init__(self):
        flux = [self.build_lookup(zip(self.ptypes, f)) for f in self.get_flux()]
        gamma = [self.build_lookup(zip(self.ptypes, g)) for g in self.get_gamma()]
        rigidity = self.get_rigidity()
        CompiledFlux.__init__(self, "+".join([self.get_expression(f, g, r) for f, g, r in zip(flux, gamma, rigidity)]))
        
class GaisserH3a(GaisserHillas):
    def get_flux(self):
        return super(GaisserH3a, self).get_flux() + [[20]*2 + [13.4]*3, [1.7]*2 + [1.14]*3]
    def get_gamma(self):
        return super(GaisserH3a, self).get_gamma() + [[2.4]*5, [2.4]*5]
    def get_rigidity(self):
        return super(GaisserH3a, self).get_rigidity() + [30*PeV, 2e3*PeV]
    
class GaisserH4a(GaisserH3a):
    def get_flux(self):
        return super(GaisserH4a, self).get_flux()[:-1] + [[200]]
    def get_gamma(self):
        return super(GaisserH4a, self).get_gamma()[:-1] + [[2.6]]
    def get_rigidity(self):
        return super(GaisserH4a, self).get_rigidity()[:-1] + [60e3*PeV]

def get_rates_corsika(cwm,nfiles):
    flux = GaisserH4a()
    pflux = flux (cwm["PrimaryEnergy"],cwm["PrimaryType"])
    energy_integral = (cwm['EnergyPrimaryMax']**(cwm["PrimarySpectralIndex"]+1)-cwm['EnergyPrimaryMin']**(cwm["PrimarySpectralIndex"]+1))/(cwm["PrimarySpectralIndex"]+1)
    energy_weight = cwm['PrimaryEnergy']**cwm["PrimarySpectralIndex"]
    w = pflux *energy_integral/energy_weight * cwm["AreaSum"] / (cwm['NEvents'] * nfiles)
    return w
