from upscaler import Upscaler
from skimage import img_as_float, img_as_ubyte
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from scipy.ndimage import filters
import numpy as np

class DCCI(Upscaler):
    """Performs Directional Cubic Convolution Interpolation,
    as defined in Zhou et al. in 
    "Image zooming using directional cubic convolution interpolation." IET Image Processing, 6 (6). 627-634"""
    def __init__(self):
        Upscaler.__init__(self,'Direct Cubic Convolution Interpolation')
    def _add_gaps(self, image):
        """Adds gaps between pixels in the given grayscale image"""
        # Add Vertical Gaps
        image = np.insert(image,range(len(image))[1:],[0],axis=0)
        # Add Horizontal Gaps
        image = np.insert(image,range(len(image[0]))[1:],[0],axis=1)
        return self._ensure_type(image)
    def _deepen(self, arr):
        """Replicates the given array on the z axis"""
        return np.dstack((arr,arr,arr))
    def _ensure_type(self,arr):
        return arr.astype(np.uint8,casting='unsafe')
    def _calc_diag_edges(self,orig_image):
        """Calculates the antidiagonal and diagonal absolute differences"""
        KERNEL_A = np.array([[0,0,1,0,1,0,1],
                             [0,0,0,0,0,0,0],
                             [0,0,1,0,1,0,1],
                             [0,0,0,0,0,0,0],
                             [0,0,1,0,1,0,1],
                             [0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0]],dtype=np.int32)
        KERNEL_D = np.array([[1,0,1,0,1,0,0],
                             [0,0,0,0,0,0,0],
                             [1,0,1,0,1,0,0],
                             [0,0,0,0,0,0,0],
                             [1,0,1,0,1,0,0],
                             [0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0]],dtype=np.int32)
        # Mirror over edges
        mirrored = orig_image
        mirrored = mirrored.astype(np.int32)
        mirrored = np.hstack((mirrored[:,:3][:,::-1],mirrored,mirrored[:,-3:][:,::-1]))
        mirrored = np.vstack((mirrored[:3][::-1],mirrored,mirrored[-3:][::-1]))
        antidiags = np.abs(mirrored[:-1,1:] - mirrored[1:,:-1])
        diags     = np.abs(mirrored[:-1,:-1] - mirrored[1:,1:])
        if mirrored.ndim == 3:
            antidiags = np.sum(antidiags,axis=2)
            diags     = np.sum(    diags,axis=2)
        antidiags = self._add_gaps(antidiags)
        diags     = self._add_gaps(diags)
        
        antidiags = np.hstack((np.zeros((antidiags.shape[0],2)),antidiags))
        antidiags = np.vstack((antidiags,np.zeros((2,antidiags.shape[1]))))
        diags     = np.hstack((np.zeros((diags.shape[0],2)),diags))
        diags     = np.vstack((diags,np.zeros((2,diags.shape[1]))))
        antidiags = filters.convolve(antidiags, KERNEL_A)
        diags     = filters.convolve(    diags, KERNEL_D)       
        assert antidiags.shape == diags.shape, \
            "Something went wrong. antidiags has shape %r, and diags has shape %r" % (antidiags.shape, diags.shape)
        return antidiags[6:-6,6:-6], diags[6:-6,6:-6]
    
    def _calc_diag_interps(self, orig_image):
        """Returns up-right and down-right diagonal interpolaton values for the image"""
        # W.R.T original coordinates (indices):
        # Up-Right: (-1 * [0,0] + 9 * [1,1] + 9 * [2,2] - 1 * [3,3]) / 16
        # Down-Right: (-1 * [0,3] + 9 * [1,2] + 9 * [2,1] - 1 * [3,0]) / 16
        KERNEL_UR = np.array([[ 0,0,0,0,0,0,-1],
                              [ 0,0,0,0,0,0, 0],
                              [ 0,0,0,0,9,0, 0],
                              [ 0,0,0,0,0,0, 0],
                              [ 0,0,9,0,0,0, 0],
                              [ 0,0,0,0,0,0, 0],
                              [-1,0,0,0,0,0, 0]],dtype=np.int32)
        KERNEL_DR = np.array([[-1,0,0,0,0,0, 0],
                              [ 0,0,0,0,0,0, 0],
                              [ 0,0,9,0,0,0, 0],
                              [ 0,0,0,0,0,0, 0],
                              [ 0,0,0,0,9,0, 0],
                              [ 0,0,0,0,0,0, 0],
                              [ 0,0,0,0,0,0,-1]],dtype=np.int32)
        
        @adapt_rgb(each_channel)
        def _do_conv_ur(image2d):
            """Convolves Up-Right (Done for Down-Right edges)"""
            return filters.convolve(image2d, KERNEL_UR)
        @adapt_rgb(each_channel)
        def _do_conv_dr(image2d):
            """Convolves Down-Right (Done for Up-Right edges)"""
            return filters.convolve(image2d, KERNEL_DR)
        
        # Mirror over edges
        mirrored = orig_image
        mirrored = self._add_gaps(mirrored).astype(np.int32,casting='safe')
        mirrored = np.hstack((mirrored[:,:3][:,::-1],mirrored,mirrored[:,-3:][:,::-1]))
        mirrored = np.vstack((mirrored[:3][::-1],mirrored,mirrored[-3:][::-1]))
        mirrored_ur = np.round(_do_conv_ur(mirrored) / 16)
        mirrored_dr = np.round(_do_conv_dr(mirrored) / 16)
        mirrored_ur = self._ensure_type(np.clip(mirrored_ur, 0, 255))
        mirrored_dr = self._ensure_type(np.clip(mirrored_dr, 0, 255))
        assert mirrored_ur.shape == mirrored_dr.shape, \
            "Something went wrong. mirrored_ur has shape %r, and mirrored_dr has shape %r" % (mirrored_ur.shape, mirrored_dr.shape)
        return mirrored_ur[3:-3,3:-3], mirrored_dr[3:-3,3:-3]
    
    def _calc_diags(self, orig_image):
        # Note: Mathematically, non-diagonal pixels should be zero
        #       in both of these arrays
        d1, d2 = self._calc_diag_edges(orig_image)
        up_right, down_right = self._calc_diag_interps(orig_image)
        # Add zeros
        expanded = self._add_gaps(orig_image)
        # Sanity checking
        assert d1.shape[:2] == expanded.shape[:2], "Shape mismatch: [d1 : %r]; [expanded : %r]" % (d1.shape, expanded.shape)
        assert up_right.shape[:2] == expanded.shape[:2], \
            "Shape mismatch: [up_right : %r]; [expanded : %r]" % (up_right.shape, expanded.shape)
        
        opd1 = 1 + d1.astype(np.int32)
        opd2 = 1 + d2.astype(np.int32)
        THRESH = 115
        up_right_edges = (100 * opd1) > (THRESH * opd2)
        down_right_edges = (100 * opd2) > (THRESH * opd1)
        # Smooth if neither up-right or down-right. We also must remove
        # non-diagonal values (since those will for sure be neither)
        orig_pixels = self._add_gaps(np.ones(orig_image.shape[:2]))
        up_right_edges *= 1 - orig_pixels
        down_right_edges *= 1 - orig_pixels
        smooth_areas = (1 - (up_right_edges + down_right_edges)) - orig_pixels
        # Interpolate smooth areas
        w1 = 1 / (1. + (np.require(d1,dtype=np.float64) ** 5))
        w2 = 1 / (1. + (np.require(d2,dtype=np.float64) ** 5))
        w3 = w1 + w2
        weight1 = self._deepen(w1 / w3)
        weight2 = self._deepen(w2 / w3)
        
        smooth_interp = self._ensure_type(np.clip(np.round((down_right * weight1) + (up_right * weight2)),0,255))
        
        #print "D1/D2:"
        #print d1[:15,:15]
        #print d2[:15,:15]
        #print "Weight1 & Down Right"
        #print weight1[:15,:15,0]
        #print down_right[:15,:15]
        #print "Weight2 & Up Right"
        #print weight2[:15,:15,0]
        #print up_right[:15,:15]
        #print "Smooth Interp"
        #print smooth_interp[:15,:15]
        
        # Note to self: might get mad here since edge matrices are 2d & [up|down]_right are 3d
        # Interpolate in opposite direction of edge
        up_right   *= self._deepen(down_right_edges)
        down_right *=   self._deepen(up_right_edges)
        smooth_interp *=  self._deepen(smooth_areas)
        expanded += up_right + down_right + smooth_interp
        return self._ensure_type(expanded)
    
    def _calc_hv_edges(self, partial_image):
        """Calculates the edge strengths in the horizontal and vertical directions"""
        H_KERNEL = np.array([[0,0,0,0,0,0,0],
                             [0,0,0,0,1,0,0],
                             [0,0,0,1,0,1,0],
                             [0,0,1,0,1,0,1],
                             [0,0,0,1,0,1,0],
                             [0,0,0,0,1,0,0],
                             [0,0,0,0,0,0,0]],dtype=np.int32)
        V_KERNEL = np.array([[0,0,0,1,0,0,0],
                             [0,0,1,0,1,0,0],
                             [0,1,0,1,0,1,0],
                             [0,0,1,0,1,0,0],
                             [0,0,0,1,0,0,0],
                             [0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0]],dtype=np.int32)
        
        mirrored = partial_image
        mirrored = mirrored.astype(np.int32)
        mirrored = np.hstack((mirrored[:,:3][:,::-1],mirrored,mirrored[:,-3:][:,::-1]))
        mirrored = np.vstack((mirrored[:3][::-1],mirrored,mirrored[-3:][::-1]))
        # Will hold differences
        h_diffs = np.zeros((mirrored.shape[0],mirrored.shape[1]-2) + mirrored.shape[2:])
        v_diffs = np.zeros((mirrored.shape[0]-2,) + mirrored.shape[1:])
        # A little difficult to read, but calculates edge strengths using
        # horizontal and vertical differences
        h_diffs[::2,::2] = np.abs(mirrored[ ::2,  2::2] - mirrored[ ::2, :-1:2])
        h_diffs[1::2,1::2] = np.abs(mirrored[1::2,  3::2] - mirrored[1::2,  1:-2:2])
        v_diffs[::2,::2] = np.abs(mirrored[ :-1:2, ::2] - mirrored[2::2,  ::2])
        v_diffs[1::2,1::2] = np.abs(mirrored[1:-2:2, 1::2] - mirrored[3::2,  1::2])
        if mirrored.ndim == 3:
            h_diffs = np.sum(h_diffs,axis=2)
            v_diffs = np.sum(v_diffs,axis=2)
        h_diffs = np.hstack((np.zeros((h_diffs.shape[0],2)),h_diffs))
        v_diffs = np.vstack((v_diffs,np.zeros((2,v_diffs.shape[1]))))
        h_diffs = filters.convolve(h_diffs, H_KERNEL)
        v_diffs = filters.convolve(v_diffs, V_KERNEL)
        assert h_diffs.shape == v_diffs.shape, \
            "Something went wrong. h_diffs has shape %r, and v_diffs has shape %r" % (h_diffs.shape, v_diffs.shape)
        return h_diffs[3:-3,3:-3], v_diffs[3:-3,3:-3]
    
    def _calc_hv_interps(self, partial_image):
        """Returns horizontal and vertical interpolaton values for the image"""
        # ([X,Y] are indices)
        # Horizontal: (-1 * [3,0] + 9 * [3,2] + 9 * [3,4] - 1 * [3,6]) / 16
        # Vertical: (-1 * [0,3] + 9 * [2,3] + 9 * [4,3] - 1 * [6,3]) / 16
        KERNEL_H = np.array([[ 0,0,0,0,0,0, 0],
                             [ 0,0,0,0,0,0, 0],
                             [ 0,0,0,0,0,0, 0],
                             [-1,0,9,0,9,0,-1],
                             [ 0,0,0,0,0,0, 0],
                             [ 0,0,0,0,0,0, 0],
                             [ 0,0,0,0,0,0, 0]],dtype=np.int32)
        KERNEL_V = np.array([[0,0,0,-1,0,0,0],
                             [0,0,0, 0,0,0,0],
                             [0,0,0, 9,0,0,0],
                             [0,0,0, 0,0,0,0],
                             [0,0,0, 9,0,0,0],
                             [0,0,0, 0,0,0,0],
                             [0,0,0,-1,0,0,0]],dtype=np.int32)
        
        @adapt_rgb(each_channel)
        def _do_conv_h(image2d):
            """Convolves Horizontally (Done for Vertical edges)"""
            return filters.convolve(image2d, KERNEL_H)
        @adapt_rgb(each_channel)
        def _do_conv_v(image2d):
            """Convolves Vertically (Done for Horizontal edges)"""
            return filters.convolve(image2d, KERNEL_V)
        
        # Mirror over edges
        mirrored = partial_image
        mirrored = np.hstack((mirrored[:,:3][:,::-1],mirrored,mirrored[:,-3:][:,::-1]))
        mirrored = np.vstack((mirrored[:3][::-1],mirrored,mirrored[-3:][::-1]))
        mirrored = mirrored.astype(np.int32,casting='safe')
        
        mirrored_h = np.round(_do_conv_h(mirrored) / 16)
        mirrored_v = np.round(_do_conv_v(mirrored) / 16)
        mirrored_h = self._ensure_type(np.clip(mirrored_h, 0, 255))
        mirrored_v = self._ensure_type(np.clip(mirrored_v, 0, 255))
        assert mirrored_h.shape == mirrored_v.shape, \
            "Something went wrong. mirrored_h has shape %r, and mirrored_v has shape %r" % (mirrored_h.shape, mirrored_v.shape)
        return mirrored_h[3:-3,3:-3], mirrored_v[3:-3,3:-3]
    
    def _calc_hv(self, partial_image):
        # Note: Mathematically, non-diagonal pixels should be zero
        #       in both of these arrays
        d1, d2 = self._calc_hv_edges(partial_image)
        horizontal, vertical = self._calc_hv_interps(partial_image)
        # Sanity checking
        assert d1.shape[:2] == partial_image.shape[:2], \
            "Shape mismatch: [d1 : %r]; [partial_image : %r]" % (d1.shape, partial_image.shape)
        assert horizontal.shape[:2] == partial_image.shape[:2], \
            "Shape mismatch: [horizontal : %r]; [expanded : %r]" % (horizontal.shape, partial_image.shape)
        
        THRESH = 115
        opd1 = 1 + d1.astype(np.int32)
        opd2 = 1 + d2.astype(np.int32)
        horizontal_edges = (100 * opd1) > (THRESH * opd2)
        vertical_edges = (100 * opd2) > (THRESH * opd1)
        # Smooth if neither up-right or down-right. We also should remove
        # values at positions of original pixels
        orig_pixels = self._add_gaps(np.ones(((partial_image.shape[0] / 2)+1, (partial_image.shape[1] / 2)+1)))
        horizontal_edges *= 1 - orig_pixels
        vertical_edges *= 1 - orig_pixels
        smooth_areas = (1 - (horizontal_edges + vertical_edges)) - orig_pixels
        
        # Interpolate smooth areas
        w1 = 1 / (1. + (np.require(d1,dtype=np.float64) ** 5))
        w2 = 1 / (1. + (np.require(d2,dtype=np.float64) ** 5))
        w3 = w1 + w2
        weight1 = self._deepen(w1 / w3)
        weight2 = self._deepen(w2 / w3)
        smooth_interp = np.clip(np.round((vertical * weight1) + (horizontal * weight2)),0,255)
        
        # Note to self: might get mad here since edge matrices are 2d & [up|down]_right are 3d
        # Interpolate in opposite direction of edge
        horizontal   *= self._deepen(horizontal_edges)
        vertical *=   self._deepen(vertical_edges)
        smooth_interp *=  self._deepen(smooth_areas)
        return self._ensure_type(np.round(np.clip(partial_image + horizontal + vertical + smooth_interp,0,255)))
        
    def _do_upscale(self,image,times):
        if times == 1:
            # Copy to keep result pure
            return np.copy(image)
        assert (times & (times - 1)) == 0, "times must be a power of two: %r" % times
        oddWidth = image.shape[1] % 2 == 1
        image = img_as_ubyte(image)
        image = np.hstack((image[:,:3][:,::-1],image,image[:,-3:][:,::-1]))
        image = np.vstack((image[:3][::-1],image,image[-3:][::-1]))
        upscaled = self._calc_hv(self._calc_diags(image))
        if oddWidth:
            upscaled = upscaled[6:-6,5:-5]
        else:
            upscaled = upscaled[6:-6,6:-6]
        # Skip unneeded copying:
        if times == 2:
            return upscaled
        return self._do_upscale(upscaled, times / 2)
