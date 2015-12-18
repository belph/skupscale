from upscaler import Upscaler
from scipy.ndimage import interpolation
import numpy as np

class Bilinear(Upscaler):
    def __init__(self):
        Upscaler.__init__(self,'Bilinear',grayscale=Upscaler.EachChannel)
    
    def _do_upscale(self,image,times):
        assert times > 0, "Times must be a number greater than or equal to one: %r" % times
        if times == 1:
            return image
        new_shape = np.asarray(image.shape)
        new_shape[0] *= times
        new_shape[1] *= times
        indices = np.asarray(np.indices(new_shape),dtype=np.float64)
        indices /= times
        return interpolation.map_coordinates(image,indices,order=1)
