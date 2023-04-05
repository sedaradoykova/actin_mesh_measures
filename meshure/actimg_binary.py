import os, json
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
        self.estimated_parameters = {'cell_surface_area': None,
                                     'mesh_density': None, 
                                     'mesh_holes': None,
                                     'cell_type': None}
        self.cell_surface_area, self.mesh_holes = None, None
        self.log = ''



    def _get_contour(self, img_in=None, n_dilations: int=3, closing_structure: any=None, return_outpt: bool=False):
        """ Returns the longest contour as a mask, all contours detected, and the index of the longest contour. 
        Note: close and dilate image serially to ensure robustness to small gaps and inconsistencies in cell boundary;
        find contours and the index of the longest contour; map contours onto new mask. 
        Default behaviour: copy binary mesh and get largest contour = cell surface. 
        Note: there is a 1 pixel additional margin because get_boundaries returns a float index (half a pixel is permissible); 
        therefore, the boundary is two-pixels thick, rounding up and down. 
        ...
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
        if img_in is None: 
            img = np.copy(self.binary_mesh)
        else: 
            img = img_in
        # close
        closed_image = binary_closing(img, structure=closing_structure)
        # dilate serially
        dilated = ski_morph.dilation(closed_image).astype('int')
        for n in np.arange(n_dilations-1):
            dilated = ski_morph.dilation(dilated).astype('int')
        # fill
        filled_image = binary_fill_holes(dilated)
        # find all contours and select longest
        contours = measure.find_contours(filled_image)
        ind_max = np.argmax([x.shape[0] for x in contours])
        contour_ceil = np.ceil(contours[ind_max]).astype('int')
        contour_floor = np.floor(contours[ind_max]).astype('int')

        contour_img = np.zeros(filled_image.shape)
        contour_img[contour_ceil[:,0], contour_ceil[:,1]] = 1
        contour_img[contour_floor[:,0], contour_floor[:,1]] = 1

        if img_in is None: 
            self.contour_img, self.contours, self.ind_max = contour_img, contours, ind_max
            self.contour_ceil, self.contour_floor = contour_ceil, contour_floor
        if return_outpt: 
            return contour_img, contours, (contour_ceil, contour_floor), ind_max
        else:
            return None


    def _fill_contour_img(self, img_in=None, n_erosions: int=4, extra_dilate_fill: bool=True, return_outpt: bool=False):
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
            A mask with the filled and thinned cell surface; mapped to filled_contour_img attribute. 
        See also
        --------
        `_get_contour()`, `_check_boundaries()`, `surface_area()`
        """
        if img_in is None: 
            img = np.copy(self.contour_img)
        else: 
            img = np.copy(img_in)

        if extra_dilate_fill:
            # dilate contour 
            dilated_cont = ski_morph.dilation(img).astype('int')
            # fill contour 
            filled_cont = binary_fill_holes(dilated_cont).astype('int')
        else: 
            filled_cont = binary_fill_holes(img).astype('int')
        # erode serially (equal to no. dilations)
        eroded_contour_img = ski_morph.binary_erosion(filled_cont).astype('int')
        for n in np.arange(n_erosions):
            eroded_contour_img = ski_morph.binary_erosion(eroded_contour_img).astype('int')
        
        if img_in is None: 
            self.filled_contour_img = eroded_contour_img
        if return_outpt: 
            return eroded_contour_img
        else:
            return None


    def _check_boundaries(self, img_in=None, contour_in=None, return_outpt: bool=False):
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
        if img_in is None: 
            contour_img = np.copy(self.contour_img)
            contour_floor = np.copy(self.contour_floor)
            up_r, up_c = self.shape
        else:
            contour_img = np.copy(img_in)
            contour_floor = np.copy(contour_in)
            up_r, up_c = img_in.shape
        # check left boundary 
        # contour_floor, contour_ceil  --  currently uses only floor, add ceil if necessary
        out = np.unique(np.array([r for r,c in contour_floor if c == 0]))
        if out.shape[0] == 1: 
            contour_img[out,0] = 1
        elif (out.shape[0] > 1) and (out.shape[0] < 6): 
            contour_img[np.min(out):np.max(out),0] = 1
        elif out.shape[0] == 0:
            pass
        else:
            print('Left: confusing boundary case; too many broken up boundaries along axis.')
        # check right boundary 
        out = np.unique(np.array([r for r,c in contour_floor if c == up_c]))
        if out.shape[0] == 1: 
            contour_img[out,up_c] = 1
        elif (out.shape[0] > 1) and (out.shape[0] < 6): 
            contour_img[np.min(out):np.max(out),up_c] = 1
        elif out.shape[0] == 0:
            pass
        else:
            print('Right: confusing boundary case; too many broken up boundaries along axis.')
        # check lower boundary  
        out = np.unique(np.array([c for r,c in contour_floor if r == 0]))
        if out.shape[0] == 1: 
            contour_img[0,out] = 1
        elif (out.shape[0] > 1) and (out.shape[0] < 6): 
            contour_img[0, np.min(out):np.max(out)] = 1
        elif out.shape[0] == 0:
            pass
        else:
            print('Bottom: confusing boundary case; too many broken up boundaries along axis.')
        # check upper boundary 
        out = np.unique(np.array([c for r,c in contour_floor if r == up_r]))
        if out.shape[0] == 1: 
            contour_img[up_r,out] = 1
        elif (out.shape[0] > 1) and (out.shape[0] < 6): 
            contour_img[up_r,np.min(out):np.max(out)] = 1
        elif out.shape[0] == 0:
            pass
        else:
            print('Top: confusing boundary case; too many broken up boundaries along axis.')
        
        if img_in is None: 
            self.contour_img = contour_img
        if return_outpt: 
            return contour_img 
        else:
            return None


    def _clean_up_surface(self, close_very_small_holes: bool=True, return_outpt: bool=False):
        """ Returns a cell surface that has been cleaned up with dilation/erosion artifacts. 
        Arguments
        ---------
        """
        # filled contour cooridnates 
        inside_coords = np.nonzero(self.filled_contour_img)
        # create a mask of the cell outline and contained mesh originally segmented 
        mesh_outline = np.zeros(self.shape)
        mesh_outline[inside_coords] = self.binary_mesh[inside_coords]
        # repeat cell outline because it's not segmented initially  
        mesh_outline[np.nonzero(self.contour_img)] = 1
        if close_very_small_holes:
            mesh_outline = binary_closing(mesh_outline, structure = np.ones((1,2)), border_value=1)
            mesh_outline = binary_closing(mesh_outline, structure = np.ones((2,1)), border_value=1)

        self.mesh_outline = mesh_outline.astype('int')
        self.mesh_inverted = (self.mesh_outline==0).astype('int')

        labels = measure.label(self.mesh_inverted, connectivity=1) # 4-connectivity for 2d images

        # shift the contour by a pixel in 8 orthogonal directions
        # if label overlaps with contour, mark to remove it potentially 
        labs_to_rm = []
        labs_outside = []
        for r_plus, c_plus in zip((1,-1,0,0,1,-1,1,-1), (0,0,1,-1,1,-1,-1,1)):
            try: 
                newlabs = [*np.unique(labels[self.contour_ceil[:,0]+r_plus, self.contour_ceil[:,1]+c_plus])]
                labs_to_rm += newlabs
            except IndexError: 
                pass
        rm_inds = np.unique(np.asarray(labs_to_rm))
        # remove labels in rm_inds only if they are smaller than 20 px
        for labval in rm_inds: 
            labinds = np.where(labels==labval) 
            if len(labinds[0]) <=20:
                labels[labinds] = 0
        # check if any labels lie outside of the cell outline and remove them          
        for labval in np.unique(labels): 
            labinds = np.where(labels==labval)
            outline_coords = np.nonzero(self.mesh_outline)
            for n in range(2):
                if ( ( (np.min(labinds[n]) < np.min(outline_coords[n])) or (np.max(labinds[n]) > np.max(outline_coords[n])) ) and 
                    ( (0 in labinds[n]) or (self.shape[n] in labinds[n]) ) ):
                    labs_outside += newlabs
        rm_inds_outside = np.unique(np.asarray(labs_outside))
        for labval in rm_inds_outside: 
            labinds = np.where(labels==labval) 
            labels[labinds] = 0

        self.labels = labels
        if return_outpt:
            return labels
        else:
            return None


    def surface_area(self, n_dilations_erosions: tuple=(3,4), closing_structure: bool=None, extra_dilate_fill: bool=True,
                     verbose: bool=False, return_outpt: bool=False):
        """ 
        Returns the surface area and associated units. 
        Note: the algorithm uses serial dilations to include the periphery of the cell even if the cell boundary is discontinuous.
        The dilations are followed by an equivalent number of serial erosions to avoid overestimating the cell area. 
        Note: it is assumed that the largest object in the field of view is the cell which is to be segmented.
        Returns
        -------
        if the cell is not segmented, this is recorded separately

        """
        if len(n_dilations_erosions) == 2: 
            n_dilations, n_erosions = n_dilations_erosions 
        elif len((n_dilations_erosions)) == 1: 
            n_dilations = n_erosions = n_dilations_erosions
        else:
            raise ValueError(f'Invalid input {n_dilations_erosions}; `n_dilations_erosions` must be an int or tuple of len=2. ')
        self._get_contour(img_in=None, n_dilations=n_dilations, closing_structure=closing_structure)
        self._fill_contour_img(img_in=None, n_erosions=n_erosions, extra_dilate_fill=extra_dilate_fill)
        
        cell_surface_area = np.sum(self.filled_contour_img)*(self.resolution['pixel_size_xy']**2)/1e6
        if verbose:
            print(f'Surface area (um^2)  =  {cell_surface_area:.2f}')
        if cell_surface_area/(self.shape[0]*self.shape[1]) > 0.7:
            msg = f'{self.title}: surface area too large. Inspect manually. \n\n'
            if verbose: 
                print(msg)
            else:
                self.log += msg

        elif cell_surface_area < np.sum(self.binary_mesh)*(self.resolution['pixel_size_xy']**2)/1e6 - cell_surface_area:
            msg = f'{self.title} segmented surface area too small. Checking if it touches boundaries. \n\n'
            if verbose:  
                print(msg)
            else:
                self.log += msg

            self._check_boundaries()
            self._fill_contour_img(n_erosions=n_erosions, extra_dilate_fill=extra_dilate_fill)
            
            cell_surface_area = np.sum(self.filled_contour_img)*(self.resolution['pixel_size_xy']**2)/1e6
            if verbose: 
                print(f'Surface area (um^2)  =  {cell_surface_area:.2f}')

            if cell_surface_area < np.sum(self.binary_mesh)*(self.resolution['pixel_size_xy']**2)/1e6 - cell_surface_area:
                msg = f'{self.title}: surface area too small despite filing edges. Inspect manually. \n\n'
                if verbose: 
                    print(msg)
                else: 
                    self.log += msg
                raise RuntimeError('Surface area not segmented.')
            elif cell_surface_area/(self.shape[0]*self.shape[1]) > 0.7:
                msg = f'{self.title}: surface area too large after filing edges. Inspect manually. \n\n'
                if verbose: 
                    print(msg)
                else: 
                    self.log += msg
                raise RuntimeError('Surface area not segmented.')
            else:
                self.cell_surface_area = cell_surface_area
                self.estimated_parameters['cell_surface_area'] = {'area': cell_surface_area, 'unit': 'um^2'}
                if return_outpt:
                    return cell_surface_area
                else:
                    return None

        else: 
            self.cell_surface_area = cell_surface_area
            self.estimated_parameters['cell_surface_area'] = {'area': cell_surface_area, 'unit': 'um^2'}
            if return_outpt:
                return cell_surface_area
            else:
                return None


    def mesh_density(self, verbose: bool=False, return_outpt: bool=False):
        """Returns the mesh density as a % value. This is defined as the percentage of the mesh / filled_cell_surface
        Note: the outline is included in the calculation. 
        This does not necessarily match the image signal in the case of discontinuous cell surfaces. 
        """
        if not self.cell_surface_area:
            raise RuntimeError('Cannot estimate mesh density because cell surface is not / cannot be segmented.')
        mesh_density = np.sum(self.mesh_outline)*100 / np.sum(self.filled_contour_img)
        self.estimated_parameters['mesh_density'] = mesh_density
        if verbose: 
            print(f'The percentage mesh density is  {mesh_density:.2f} %')
            print('Defined as the difference between the filled and unfilled mask.')
        if return_outpt:
            return mesh_density
        else: 
            return None


    def mesh_holes_area(self, unit: str='um', saturation_area: float=1, visualise: bool=False, return_outp: bool=False):
        """ Returns the labels by area and visualises mesh holes coloured by area size with specified unit and saturation 
        """
        if not self.cell_surface_area:
            try: 
                self.mesh_holes_area()
            except RuntimeError: 
                raise AttributeError('Mesh holes cannot be quantified because cell surface has not been segmented.')

        self._clean_up_surface(close_very_small_holes=True, return_outpt=False)
        # label every region by area  
        f_labels_area = np.zeros(self.shape).astype('float')
        if unit == 'nm':
            area_factor = (self.resolution['pixel_size_xy'])**2 if self.resolution['unit'] == 'nm' else (self.resolution['pixel_size_xy']*1e-3)**2
        elif unit == 'um':
             area_factor = (self.resolution['pixel_size_xy']*1e-3)**2 if self.resolution['unit'] == 'nm' else (self.resolution['pixel_size_xy'])**2
        elif unit == 'px':
            area_factor=1
        for labval in np.unique(self.labels): 
            labinds = np.where(self.labels==labval) 
            area = len(labinds[0])*area_factor  
            f_labels_area[labinds] = area if ((labval != 1) and (labval != 0)) else 0

        # make transparent mesh and labels ready for visualisation 
        mesh_contour_transparent = self.mesh_outline.copy().astype('float')
        mesh_contour_transparent[np.where(np.isclose(mesh_contour_transparent, 1))] = np.nan

        f_labels_transparent = f_labels_area.copy()
        f_labels_transparent[np.where(np.isclose(f_labels_transparent, 0))] = np.nan
        if saturation_area is not None: 
            f_labels_transparent[f_labels_transparent >= saturation_area] = saturation_area
            self.__saturation_area = saturation_area if saturation_area else None

        if visualise:
            plt.imshow(mesh_contour_transparent, cmap='gray')
            plt.imshow(f_labels_transparent, cmap='coolwarm_r')
            colbar = plt.colorbar()
            if saturation_area is not None:
                new_ticks = np.linspace(0, saturation_area, 6)
                new_ticklabs = [f'{n:.2e}' for n in new_ticks]
                new_ticklabs[-1] = f'>={saturation_area:.2e}'
                new_ticklabs[0] = '0'
                colbar.set_ticks(new_ticks)
                colbar.set_ticklabels(new_ticklabs)
            plt.title(f'Area of mesh holes ({unit}$^2$)')
            plt.axis('off'); plt.tight_layout(); plt.show()

        self.f_labels_area, self.mesh_holes = f_labels_area, {'segmented': True, 'area_factor': area_factor, 'unit': f'{unit}^2'}
        self._mesh_contour_transparent, self._f_labels_transparent = mesh_contour_transparent, f_labels_transparent
        if return_outp:
            return f_labels_area
        else:
            return None
        
    def visualise_segmentation(self, save: bool=False, dest_dir: str=os.getcwd()):
        """ Visualise the segmentation steps: binary mesh, cell surface, mesh holes. 
        """
        if not isinstance(save, bool):
            raise TypeError('`save` must be a boolean.')
        if save and (not os.path.exists(dest_dir)):
            raise FileExistsError(f'Path not found: {dest_dir}')
        if not self.cell_surface_area:
            try: 
                area = self.surface_area(return_outpt=True)
                self.cell_surface_area = area
                self.estimated_parameters['cell_surface_area'] = {'area': area, 'unit': 'um^2'}
            except RuntimeError: 
                raise RuntimeError('Mesh cannot be segmented as surface area segmentation failed.')
        if not self.mesh_holes:
            try: 
                self.surface_area()
                self.mesh_holes_area()
            except: 
                raise RuntimeError('Mesh holes have not been segmented. Call the `surface_area()` and `mesh_holes_area()` methods.')

        plt.figure(figsize=(14,5))
        plt.subplot(1,3,1)
        plt.title('Segmented mesh')
        plt.imshow(self.binary_mesh, cmap='gray')
        plt.axis('off')
        plt.subplot(1,3,2)
        ar, un = self.estimated_parameters["cell_surface_area"].values()
        plt.title(f'Segmented cell surface ({ar:.2f} {un})')
        plt.imshow(self.mesh_outline, cmap='gray')
        plt.imshow(self.filled_contour_img, cmap='gray', alpha=0.7)
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.title(f'Segmented mesh holes ({self.mesh_holes["unit"]})')
        plt.imshow(self._mesh_contour_transparent, cmap='gray')
        plt.imshow(self._f_labels_transparent, cmap='coolwarm_r')
        colbar = plt.colorbar(fraction=0.05, pad=0.01)
        if self.__saturation_area is not None:
            new_ticks = np.linspace(0, self.__saturation_area, 6)
            new_ticklabs = [f'{n:.2e}' for n in new_ticks]
            new_ticklabs[-1] = f'>={self.__saturation_area:.2e}'
            new_ticklabs[0] = '0'
            colbar.set_ticks(new_ticks)
            colbar.set_ticklabels(new_ticklabs)
        plt.axis('off')
        plt.tight_layout()
        if save: 
            title = f'{self.title.split(".")[0]}_mesh_segmentation.png'
            plt.savefig(os.path.join(dest_dir, title), dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()
    
    def quantify_mesh(self):
        if not self.cell_surface_area:
            try: 
                area = self.surface_area(return_outpt=True)
                self.cell_surface_area = area
                self.estimated_parameters['cell_surface_area'] = {'area': area, 'unit': 'um^2'}
            except RuntimeError: 
                raise RuntimeError('Mesh cannot be segmented as surface area segmentation failed.')
        if not self.mesh_holes:
            try: 
                self.mesh_holes_area()
            except RuntimeError: 
                raise RuntimeError('Mesh holes cannot be segmented.')
        hole_parameters = pd.DataFrame(measure.regionprops_table(self.labels, self.binary_mesh, properties=['equivalent_diameter_area', 'area', 'perimeter']))
        hole_parameters.equivalent_diameter_area = hole_parameters.equivalent_diameter_area*self.mesh_holes['area_factor']
        hole_parameters.area = hole_parameters.area*self.mesh_holes['area_factor']
        hole_parameters.perimeter = hole_parameters.perimeter*np.sqrt(self.mesh_holes['area_factor'])
        hole_parameters = hole_parameters.rename(columns={
            'equivalent_diameter_area': f'equivalent_diameter_area_{self.mesh_holes["unit"]}',
            'area': f'area_{self.mesh_holes["unit"]}', 'perimeter': f'perimeter_{self.mesh_holes["unit"].split("^")[0]}' })

        self.estimated_parameters['mesh_holes'] = {'hole_parameters': hole_parameters, 'unit': self.mesh_holes["unit"]}
        return None

    def _get_activation_time(self):
        activation_time = [char for char in self.title.split('_') if 'min' in char][0] 
        return activation_time

    def save_estimated_parameters(self, dest_dir: str=os.getcwd()): 
        if not os.path.exists(dest_dir):
            raise FileExistsError(f'Path not found: {dest_dir}')
        
        dest = os.path.join(dest_dir, 'params')
        if not os.path.exists(dest):
            os.mkdir(dest)

        if 'activation_time' not in self.estimated_parameters['cell_type'].keys():
            self.estimated_parameters['cell_type']['activation_time'] = self._get_activation_time()

        try:
            json_params = json.dumps({'filename': self.title,
                                      'cell_type': self.estimated_parameters['cell_type'],
                                    'resolution': self.resolution,
                                    'cell_surface_area': self.estimated_parameters['cell_surface_area'],
                                    'mesh_density': self.estimated_parameters['mesh_density']})
            file = 'params_'+self.title.split(".")[0]+'.json' 
            with open(os.path.join(dest, file), 'w') as f:
                f.write(json_params)
            
            self.estimated_parameters['mesh_holes']['hole_parameters'].to_csv(
                os.path.join(dest, 'params_'+self.title.split(".")[0]+'.csv'),sep=',',index=False,header=True)
        except: 
            raise RuntimeError('There was a problem writing the parameters. See log and check pipeline.')
        return None


    def save_log(self,dest):
        if len(self.log) > 0:
            self.log = self.title + '\n\n' + self.log
            with open(os.path.join(dest, f'{self.title.split(".")[0]}_log.txt')) as f:
                f.write(self.log)
        return None

def get_ActImgBinary(actimg): 
    """ Returns an ActImgBinary instance given a related ActImg. 
    Arguments
    ---------
    actimg : ActImg
        An instance of the ActImg class.
    """
    if not isinstance(actimg, ActImg):
        raise TypeError('Input must be an ActImg instance.')
    
    actimgbinary = ActImgBinary(binary_mesh=actimg.manipulated_stack, image_stack=actimg.image_stack, title=actimg.title, 
                                shape=actimg.shape, depth=actimg.depth, deconvolved=actimg.deconvolved, resolution=actimg.resolution,
                                meta=actimg.meta)
    return actimgbinary

