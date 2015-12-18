from skimage import color
from skimage.color.adapt_rgb import adapt_rgb, each_channel
import matplotlib.image as mpimg
class Upscaler:
    """Abstract base class for upscaling algorithms"""
    Convert, EachChannel = range(2)
    def __init__(self, name, grayscale=None):
        self.name = name
        assert grayscale is None or \
            grayscale is Upscaler.Convert or \
            grayscale is Upscaler.EachChannel, \
            "grayscale must be one of None, Convert, or EachChannel; Given: %r" % grayscale
            
        self.grayscale = grayscale
        
    def _do_upscale(self,*args,**kwargs):
        raise NotImplementedError("Upscaler " + self.name + " must implement method _do_upscale()")
        
    def _do_upscale_rgb(self,image,*args,**kwargs):
        img_is_grayscale = image.ndim == 2
        if img_is_grayscale:
            return self._do_upscale(color.gray2rgb(image),*args,**kwargs)
        else:
            return self._do_upscale(image,*args,**kwargs)
        
    def _do_upscale_gray(self,image,*args,**kwargs):
        img_is_grayscale = image.ndim == 2
        
        @adapt_rgb(each_channel)
        def adapted(image,*args,**kwargs): return self._do_upscale(image,*args,**kwargs)
        
        if img_is_grayscale:
            return self._do_upscale(image, *args, **kwargs)
        elif self.grayscale is Upscaler.Convert:
            return self._do_upscale(color.rgb2gray(image),*args,**kwargs)
        elif self.grayscale is Upscaler.EachChannel:
            return adapted(image,*args,**kwargs)
        
    def __call__(self,image,*args,**kwargs):
        if self.grayscale:
            return self._do_upscale_gray(image,*args,**kwargs)
        else:
            return self._do_upscale_rgb(image,*args,**kwargs)
        
    @classmethod
    def upscale(cls, image, *args, **kwargs):
        inst = cls()
        return inst(image,*args,**kwargs)
    @classmethod
    def scale_and_save(cls,image,savename,*args,**kwargs):
        inst = cls()
        mpimg.imsave(savename,inst(image,*args,**kwargs))
