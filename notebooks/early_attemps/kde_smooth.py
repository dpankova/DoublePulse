import numpy as np
from KDEpy import FFTKDE

def kde_smooth(a):
    '''
    Smoothing a 2d array of waveforms along the first dimension (time)
    ''' 
    x = np.arange(a.shape[0])
    # an array of x values larger than x is needed for the KDE thing
    extended_x = np.concatenate([[-1], x, [a.shape[0]]])
    
    out = []
    for i, z_idx in enumerate(range(a.shape[1])):
        y = a[:,z_idx]
        
        # here we will sample the waveform, creating a number of points
        # depending on the ADC count, higher ADC -> more points
        points = []
        weights = []
        for idx in range(len(x)):
            n = int(np.ceil(y[idx]**0.7 * 0.1))
            points.extend(x[idx] + np.random.rand(n) -0.5 )
            weights.extend([n**0.5]*n)
        points = np.array(points)
        weights = np.array(weights)

        # this is the actual KDE
        yf = FFTKDE(bw='ISJ').fit(points, weights=weights)(extended_x)
        # cut away the extended values, and normalize
        out.append(yf[1:-1] * np.sum(y))
    
    # now patch it back together into a 2d array
    return np.stack(out).T
