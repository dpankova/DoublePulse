
"""
Custom axes for ternary projection, roughly adapted and updated from
ternary_project.py by Kevin L. Davies 
"""

import matplotlib
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Polygon
from matplotlib.path import Path
from matplotlib.ticker import NullLocator, Formatter, FixedLocator
from matplotlib.transforms import Affine2D, IdentityTransform, BboxTransformTo, Transform
from matplotlib.projections import register_projection
import matplotlib.spines as mspines
import matplotlib.axis as maxis

import numpy as np

class TernaryAxes(Axes):
    name = 'ternary1'
    angle = 0.
    def __init__(self, *args, **kwargs):
        Axes.__init__(self, *args, **kwargs)
        self.cla()
        self.set_aspect(aspect='equal', adjustable='box', anchor='C') # C is for center.
    
    def cla(self):
        super(TernaryAxes, self).cla()
        
        self.set_xlim(0, 1)
        self.set_ylim(0, 1)
        self.yaxis.set_visible(False)
        self.xaxis.set_ticks_position('bottom')
        self.xaxis.set_ticks(np.linspace(0, 1, 11))
        
        self.xaxis.set_label_coords(0.5, 0, transform=self._xlabel_transform)
    
    @classmethod
    def create(cls, fig=None, subplotspec=111):
        """
        Top-level factory method. Use this to create new axes.
        """
        if fig is None:
            import pylab
            fig = pylab.gcf()
        self = fig.add_subplot(subplotspec, projection="ternary1")
        self.ab = self
        self.bc = self.figure.add_axes(self.get_position(True), sharex=self,
                                        projection="ternary2", frameon=True)
        self.ca = self.figure.add_axes(self.get_position(True), sharex=self,
                                        projection="ternary3", frameon=True)
        return self
    
    def grid(self, *args, **kwargs):
        if not hasattr(self, 'ab'):
            super(TernaryAxes, self).grid(*args, **kwargs)
        else:
            for k in 'ab', 'bc', 'ca':
                ax = getattr(self, k)
                ax.xaxis.grid(*args, **kwargs)
    
    def _rotate(self, trans):
        pass
    
    def _set_lim_and_transforms(self):
        """
        This is called once when the plot is created to set up all the
        transforms for the data, text and grids.
        """
        # Transform from the lower triangular half of the unit square
        # to an equilateral triangle
        h = np.sqrt(3)/2
        self.transProjection = Affine2D().scale(1, h).skew_deg(30,0)

        # Shift and and rotate the axis into place
        self.transAffine = Affine2D().translate(0., 1-h)
        self._rotate(self.transAffine)

        # 3) This is the transformation from axes space to display
        # space.
        self.transAxes = BboxTransformTo(self.bbox)

        # Put all the transforms together
        self.transData = \
            self.transProjection + \
            self.transAffine + \
            self.transAxes
        
        self._xaxis_transform = self.transData
        self._yaxis_transform = self.transData
        
        # Set up a special skew-less transform for the axis labels
        self._xlabel_transform = Affine2D()
        self._rotate(self._xlabel_transform)
        self._xlabel_transform += self.transAxes
        
        # Set up a special shifted transform for the x tick labels
        pad = 20
        trans = Affine2D().translate(pad*np.sin(self.angle), -pad*np.cos(self.angle))
        #print self.name, pad*np.sin(self.angle), -pad*np.cos(self.angle)
        aff = Affine2D().translate(pad*np.sin(self.angle), -pad*np.cos(self.angle))
        # self._rotate(aff)
        self._xaxis_text1_transform = self.transData # + aff
        self._xaxis_text2_transform = IdentityTransform()
        
        
    def get_xaxis_transform(self,which='grid'):
        """
        Override this method to provide a transformation for the
        x-axis grid and ticks.
        """
        assert which in ['tick1','tick2','grid']
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pixelPad):
        """
        Override this method to provide a transformation for the
        x-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        return self._xaxis_text1_transform, 'bottom', 'center'
    
    def get_yaxis_transform(self,which='grid'):
        """
        Override this method to provide a transformation for the
        y-axis grid and ticks.
        """
        assert which in ['tick1','tick2','grid']
        return self._yaxis_transform
    
    def _gen_axes_patch(self):
        h = np.sqrt(3)/2.
        return Polygon([(0, 1-h), (1, 1-h), (0.5, 1)], closed=True)
    
    def _gen_axes_spines(self):
        path = Path([(0.0, 0.0), (1.0, 0.0)])
        return dict(bottom=mspines.Spine(self, 'bottom', path))
    
    def _init_axis(self):
        self.xaxis = maxis.XAxis(self)
        self.spines['bottom'].register_axis(self.xaxis)
        self.yaxis = maxis.YAxis(self)
        self._update_transScale()

class TernaryAxes2(TernaryAxes):
    name = "ternary2"
    angle = 2*np.pi/3
    def _rotate(self, trans):
        h = np.sqrt(3)/2
        trans.rotate_around(1, 1-h, self.angle).translate(-0.5, h)
    def cla(self):
        super(TernaryAxes2, self).cla()
        
        self.patch.set_visible(False)

class TernaryAxes3(TernaryAxes2):
    name = "ternary3"
    angle = 4*np.pi/3
    def _rotate(self, trans):
        h = np.sqrt(3)/2
        trans.rotate_around(1, 1-h, self.angle).translate(-1, 0)

register_projection(TernaryAxes)
register_projection(TernaryAxes2)
register_projection(TernaryAxes3)
