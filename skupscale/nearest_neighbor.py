from upscaler import Upscaler
import numpy as np

class NearestNeighbor(Upscaler):
    """Performs Nearest Neighbor Upscaling (i.e. tiling pixels)"""
    def __init__(self):
        Upscaler.__init__(self,'Nearest Neighbor')
    def _do_upscale(self, image, times):
        assert times > 0, "Times must be a number greater than or equal to one: %r" % times
        return np.repeat(np.repeat(image,times,axis=0),times,axis=1)
