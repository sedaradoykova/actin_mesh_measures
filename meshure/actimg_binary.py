import os 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage import measure
from skimage import morphology as ski_morph 
from meshure.actimg import ActImg



@dataclass(repr=True)
class ActImgBinary(ActImg):
    """ `ActImgBinary` is a class which helps quantify the actin mesh in binary images. 
    ...
    Attributes
    ----------
    binary_mesh : np.array=None

    Methods
    -------
    """
    binary_mesh: np.ndarray=None
    image_stack: np.ndarray=None
    shape: tuple=None
    depth: int=None
    title: str=None
    deconvolved: bool=None
    resolution: float=None
    meta: dict=None


    def __post_init__(self):
        if not (np.sort(np.unique(self.binary_mesh)) == [0,1]).all():
            raise ValueError('Input image is not binary.')
        self.contour_img, self.eroded_contour_img = None, None
        self.contours, ind_max = None, None
        self.contour_ceil, self.contour_floor = None, None
        self.labels, self.labelsfloat = None, None
        self.cell_surface_area = {'area': None, 'units': None}
        self.contour_transparent = None



    def _get_contour(self, n_dilations: int=3, closing_structure: any=None, return_outpt: bool=False):
        """ Returns the longest contour as a mask, all contours detected, and the index of the longest contour. 
        Note: close and dilate image serially to ensure robustness to small gaps and inconsistencies in cell boundary;
        find contours and the index of the longest contour; map contours onto new mask. 
        Arguments
        ---------
        n_dilations=3
            Number of serial dilations, can be adjusted to avoid introducing artifacts. 
        closing_structure : array-like, =None 
            Structure used to fill in small, presumably unwanted gaps in original segmentation. 
        Returns
        ------- 
        contour_img : np.ndarray (m, n)
            A mask with the longest contour (=1). 
        contours : list of np.ndarrays
            A list of all detected contours. 
        (contour_ceil, contour_floor) : tuple of np.ndarrays
            Rounded coordinates mapped onto contour_img.  
        ind_max : int
            The index of the longest contour in `contours`.
        """
        if closing_structure is None:
            closing_structure = np.ones((5,5))
        img = np.copy(self.binary_mesh)
        # close
        closed_image = binary_closing(img, structure=closing_structure)
        # dilate serially
        dilated = ski_morph.dilation(closed_image).astype('int')
        for n in np.arange(n_dilations-1):
            dilated = ski_morph.dilation(dilated).astype('int')
        # fill
        filled_image = binary_fill_holes(dilated)
        # find all contours and select longest
        self.contours = measure.find_contours(filled_image)
        self.ind_max = np.argmax([x.shape[0] for x in self.contours])
        self.contour_ceil = np.ceil(self.contours[self.ind_max]).astype('int')
        self.contour_floor = np.floor(self.contours[self.ind_max]).astype('int')

        self.contour_img = np.zeros(filled_image.shape)
        self.contour_img[self.contour_ceil[:,0], self.contour_ceil[:,1]] = 1
        self.contour_img[self.contour_floor[:,0], self.contour_floor[:,1]] = 1

        if return_outpt: 
            return self.contour_img, self.contours, (self.contour_ceil, self.contour_floor), self.ind_max
        else:
            return None

    def _fill_contour_img(self, n_erosions: int=4, extra_dilate_fill: bool=True, return_outpt: bool=False):
        """ Returns a filled mask of the cell surface. Uses serial erosions to avoid artifacts.
        Arguments
        ---------
        contour_img : np.ndarray 
            Binary mask of a cell contour, output of `_get_contour()`
        n_erosions : bool=4
            Number of serial erosions which prevent the creation of artifacts. 
            Should be one more than the ones introduced in `_get_contour()` for greatest degree of accuracy. 
        extra_dilate_fill : bool=True
            An optional additional dilation and filling step to capture any small areas which may be otherwise missed. 
        Returns
        -------
        eroded_contour_img : nd.array (m,n)
            A mask with the filled and thinned cell surface. 
        See also
        --------
        `_get_contour()`, `_check_boundaries()`, `surface_area()`
        """
        if extra_dilate_fill:
            # dilate contour 
            dilated_cont = ski_morph.dilation(self.contour_img).astype('int')
            # fill contour 
            filled_cont = binary_fill_holes(dilated_cont).astype('int')
        else: 
            filled_cont = binary_fill_holes(self.contour_img).astype('int')
        # erode serially (equal to no. dilations)
        eroded_contour_img = ski_morph.binary_erosion(filled_cont).astype('int')
        for n in np.arange(n_erosions):
            self.eroded_contour_img = ski_morph.binary_erosion(eroded_contour_img).astype('int')
        
        if return_outpt: 
            return eroded_contour_img
        else:
            return None


    def _check_boundaries(self, return_outpt: bool=False):
        """ Check if contour touches image boundaries. If yes, check how many points are touching a given boundary.
        If < 5 points are touching the boundary, their range will be used to fill (min, max).
        If > 5 points are touching the boundary on one axis, segmentation will fail. 
        If the surface area is too large or too small, this will be raised appropriately. 
        Arguments
        ---------
        contour_img : np.ndarray (m,n)
            A mask with the longest contour (=1); returned by `_get_contour()`.
        Returns
        -------
        log : dict
            A log detailing the kind of error expected in case of unsuccessful mesh segmentation. 
        """
        up_r, up_c = self.shape
        # check left boundary 
        # contour_floor, contour_ceil  --  currently uses only floor, add ceil if necessary
        out = np.unique(np.array([r for r,c in self.contour_floor if c == 0]))
        if out.shape[0] == 1: 
            self.contour_img[out,0] = 1
        elif (out.shape[0] > 1) and (out.shape[0] < 6): 
            self.contour_img[np.min(out):np.max(out),0] = 1
        elif out.shape[0] == 0:
            pass
        else:
            print('Left: confusing boundary case; too many broken up boundaries along axis.')
        # check right boundary 
        out = np.unique(np.array([r for r,c in self.contour_floor if c == up_c]))
        if out.shape[0] == 1: 
            self.contour_img[out,up_c] = 1
        elif (out.shape[0] > 1) and (out.shape[0] < 6): 
            self.contour_img[np.min(out):np.max(out),up_c] = 1
        elif out.shape[0] == 0:
            pass
        else:
            print('Right: confusing boundary case; too many broken up boundaries along axis.')
        # check lower boundary  
        out = np.unique(np.array([c for r,c in self.contour_floor if r == 0]))
        if out.shape[0] == 1: 
            self.contour_img[0,out] = 1
        elif (out.shape[0] > 1) and (out.shape[0] < 6): 
            self.contour_img[0, np.min(out):np.max(out)] = 1
        elif out.shape[0] == 0:
            pass
        else:
            print('Bottom: confusing boundary case; too many broken up boundaries along axis.')
        # check upper boundary 
        out = np.unique(np.array([c for r,c in self.contour_floor if r == up_r]))
        if out.shape[0] == 1: 
            self.contour_img[up_r,out] = 1
        elif (out.shape[0] > 1) and (out.shape[0] < 6): 
            self.contour_img[up_r,np.min(out):np.max(out)] = 1
        elif out.shape[0] == 0:
            pass
        else:
            print('Top: confusing boundary case; too many broken up boundaries along axis.')
        if return_outpt: 
            return self.contour_img 
        else:
            return None


    def _clean_up_surface(self, saturation_area: int=3e3, close_very_small_holes: bool=True):
        """ Returns a cell surface that has been cleaned up with dilation/erosion artifacts. 
        Arguments
        ---------
        """
        # filled contour cooridnates 
        inside_coords = np.nonzero(self.eroded_contour_img)
        # create a mask of the cell outline and contained mesh originally segmented 
        mesh_outline = np.zeros(self.shape)
        mesh_outline[inside_coords] = self.binary_mesh[inside_coords]
        # repeat cell outline because it's not segmented initially  
        mesh_outline[np.nonzero(self.contour_img)] = 1
        if close_very_small_holes:
            mesh_outline = binary_closing(mesh_outline, structure = np.ones((1,2)))
            mesh_outline = binary_closing(mesh_outline, structure = np.ones((2,1)))
            mesh_outline = self._check_boundaries(mesh_outline) #########################################

        mesh_inverted = (mesh_outline==0).astype('int')
        labels = measure.label(mesh_inverted, connectivity=1) # 4-connectivity for 2d images

        # shift the contour by a pixel in 8 orthogonal directions
        # if label overlaps with contour, remove it 
        labs_to_rm = []
        for r_plus, c_plus in zip((1,-1,0,0,1,-1,1,-1), (0,0,1,-1,1,-1,-1,1)):
            newlabs = [*np.unique(labels[self.contour_ceil[:,0]+r_plus, self.contour_ceil[:,1]+c_plus])]
            labs_to_rm += newlabs
        rm_inds = np.unique(np.asarray(labs_to_rm))

        # remove labels in rm_inds only if they are smaller than 20 px
        for labval in rm_inds: 
            labinds = np.where(labels==labval) 
            if len(labinds[0]) <=20:
                labels[labinds] = 0

        # label every region by area  
        labelsfloat = np.zeros(self.shape).astype('float')
        area_factor = self.resolution['pixel_size_xy']**2 if self.resolution['unit'] == 'nm' else 1
        for labval in np.unique(labels): 
            labinds = np.where(labels==labval) 
            newval = len(labinds[0])*area_factor  
            labelsfloat[labinds] = newval if ((labval != 1) and (labval != 0)) else 0

        # make transparent images ready for visualisation 
        labelsfloat[np.where(np.isclose(labelsfloat, 0))] = np.nan
        labelsfloat[labelsfloat >= saturation_area] = saturation_area

        contour_transparent = mesh_outline.copy().astype('float')
        contour_transparent[np.where(np.isclose(contour_transparent, 1))] = np.nan
        mesh_inverted_transp = mesh_inverted.copy().astype('float')
        mesh_inverted_transp[np.where(np.isclose(mesh_inverted_transp, 0))] = np.nan


        plt.imshow(contour_transparent, cmap='gray')
        plt.imshow(labelsfloat, cmap='coolwarm_r')
        cb = plt.colorbar()
        ticks = cb.get_ticks().astype('int').astype('str')
        ticks[-1] = f'>={ticks[-1]}'
        cb.set_ticklabels(ticks)
        plt.title('mesh hole area (nm^2)')
        plt.axis('off'); plt.tight_layout(); plt.show()

        return (labels, labelsfloat)


    def _visualise_surface_segmentation(self):
        raise NotImplementedError


    def surface_area(self, n_dilations: int=3, closing_structure: bool=None, n_erosions: int=4, extra_dilate_fill: bool=True):
        """ 
        Returns the surface area and associated units. 
        Note: the algorithm uses serial dilations to include the periphery of the cell even if the cell boundary is discontinuous.
        The dilations are followed by an equivalent number of serial erosions to avoid overestimating the cell area. 
        Note: it is assumed that the largest object in the field of view is the cell which is to be segmented.
        Returns
        -------
        if the cell is not segmented, this is recorded separately

        """
        #contour_img, contours, (contour_ceil, contour_floor), ind_max = self._get_contour(n_dilations=n_dilations, closing_structure=closing_structure)
        #eroded_contour_img = self._fill_contour_img(contour_img=contour_img, n_erosions=n_erosions, extra_dilate_fill=extra_dilate_fill)
        self._get_contour(n_dilations=n_dilations, closing_structure=closing_structure)
        self._fill_contour_img(n_erosions=n_erosions, extra_dilate_fill=extra_dilate_fill)
        
        self.cell_surface_area = np.sum(self.eroded_contour_img)*(self.resolution['pixel_size_xy']**2)/1e6
        print(f'Surface area (um^2)  =  {self.cell_surface_area:.2f}')
        if self.cell_surface_area/(self.shape[0]*self.shape[1]) > 0.7:
                print(f'{self.title}: surface area too large. Inspect manually.')

        elif self.cell_surface_area < np.sum(self.binary_mesh)*(self.resolution['pixel_size_xy']**2)/1e6 - self.cell_surface_area:
            print(f'{self.title} segmented surface area too small. Checking if it touches boundaries.')

            #contour_img = self._check_boundaries()
            self._check_boundaries()

            #eroded_contour_img = self._fill_contour_img(contour_img, n_erosions=n_erosions, extra_dilate_fill=extra_dilate_fill)
            self._fill_contour_img(n_erosions=n_erosions, extra_dilate_fill=extra_dilate_fill)
            
            self.cell_surface_area = np.sum(self.eroded_contour_img)*(self.resolution['pixel_size_xy']**2)/1e6
            print(f'Surface area (um^2)  =  {self.cell_surface_area:.2f}')

            if self.cell_surface_area < np.sum(self.binary_mesh)*(self.resolution['pixel_size_xy']**2)/1e6 - self.cell_surface_area:
                print(f'{self.title}: surface area too small despite filing edges. Inspect manually.')
            elif self.cell_surface_area/(self.shape[0]*self.shape[1]) > 0.7:
                print(f'{self.title}: surface area too large after filing edges. Inspect manually.')

        else: 
            return self.cell_surface_area


def get_ActImgBinary(actimg): 
    actimgbin = ActImgBinary(binary_mesh=actimg.manipulated_stack, image_stack=actimg.image_stack, title=actimg.title, 
                             shape=actimg.shape, depth=actimg.depth, deconvolved=actimg.deconvolved, resolution=actimg.resolution,
                             meta=actimg.meta)
    return actimgbin


from meshure.actimg import get_ActImg
data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/deconv_data/")


subs = [1]
actimg = get_ActImg('1min_FOV8_decon_bottom_left.tif ', os.path.join(data_path, 'CARs_8.11.22_processed_imageJ')) 
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0,180,20),sigma=2,substack=subs,visualise=False)
actimg.z_project_min()
actimg.threshold_dynamic(std_dev_factor=0, return_mean_std_dev=False)

actimg.meshwork_density(False)
actimg.meshwork_size(False, False, False) #FFT
actimg.surface_area(False)

actimg.visualise('manipulated')
imname, sub = '', ''

new = get_ActImgBinary(actimg)
new.visualise('original', 1)
new.surface_area(1,None,3,True)
new._clean_up_surface()