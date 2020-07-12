from __future__ import print_function, absolute_import
from .Channel25_strided import GraphEncoder_25ChanStridedDecoder
from .Channel25_upsample import GraphEncoder_25ChanUpSampleDecoder
from .RGB_upsample import GraphEncoder_RGBUpSampleDecoder
from .RGB_strided import GraphEncoder_RGBStridedDecoder
#from .Channel25_upsample_dim2688 import GraphEncoder_25ChanUpSampleDecoder_dim2000
#from .Channel25_strided_dim2688 import GraphEncoder_25ChanStridedDecoder_dim2000


__factory = {
    'strided': GraphEncoder_25ChanStridedDecoder,
    'upsample':GraphEncoder_25ChanUpSampleDecoder,
    'stridedRGB': GraphEncoder_RGBStridedDecoder,
    'upsampleRGB': GraphEncoder_RGBUpSampleDecoder,    
}


def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    """
    Create a loss instance.
    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown network:", name)
    return __factory[name](*args, **kwargs)