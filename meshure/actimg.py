import os, cv2, pickle, json, warnings, scipy.stats
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib_scalebar.scalebar import ScaleBar
from scipy import stats
from scipy.optimize import curve_fit
from scipy.ndimage import correlate, morphology, binary_closing
from skimage import measure
from skimage import morphology as ski_morph 
from skimage.measure import profile_line
from meshure.utils import get_image_stack, get_meta, get_resolution, get_fig_dims


""" TODO: 
    - [ ] Annotate steps of analysis in docstrings? or separately
    - [ ] test steerable second order filter but how?
    - [ ] steerable filter: results inconsistent with matlab (see tests/test_steer_gauss*) 
    - when to return self?   
    - [ ] finish class docstrings
    - [ ] check if the analysis method has already been called on the instance to prevent meaningless manipulations
"""

@dataclass(repr=True)
class ActImg:
    """ `ActImg` is a class which helps manipulate fluorescence microscopy images (originally designed for STED images of actin). 
    The class contains helpers to read in the data and instantiate them as `ActImg` objects (see: `get_actimg()`); 
    methods to visualise and manipulate the instances, estimating and summarising parameters relating to the actin mesh.   
    Currently, the original data is kept in the `image_stack` attribute, and modifications happen on a copy which is internally updated 
    after methods are called on it with the option of nuking (see: `actimg.nuke()`) the modifications to restore original object. 
    A history of all changes implemented (methods called) are recorded in a hidden attribute `._history`. 
    `ActImg` instances are temporarily created and manipulated by the `ActImgCollection` class.   
    ...
    Attributes
    ----------
    image_stack : tuple or np.ndarray (ndim, m, n)
        A tuple of arrays or a umtidimensional array where axis=0 specifies the frame of an image.  
    title : str
        The title of the image, also the filename.
    shape : tuple=None
    depth : int=None
    deconvolved : bool=None
    resolution : float=None
    meta : dict=None
    manipulated_stack : np.array=None
    manipulated_depth : int=0
    manipulated_substack_inds : any=None
    _projected = None
    _history = None
    estimated_parameters = {}
    _binary_images = {}
    Methods
    -------
    visualise(
        imtype: str='original', ind: int=1, save: bool=False, dest_dir: str=os.getcwd(),
        colmap: str='inferno', scale_bar: bool=True, bar_locate:str='upper left'
        ):
        Description.
    normalise()
        Description.
    ????
    """
    image_stack: tuple or np.ndarray
    title: str
    shape: tuple=None
    depth: int=None
    deconvolved: bool=None
    resolution: float=None
    meta: dict=None
    manipulated_stack: np.array=None
    manipulated_depth: int=0
    manipulated_substack_inds = None


    def __post_init__(self):
        if not isinstance(self.image_stack, np.ndarray) and not isinstance(self.image_stack, tuple):
            raise TypeError("Input data must be a numpy array or tuple of arrays.")
        if not isinstance(self.title, str):
            raise TypeError("Title must be a string.")
        if not isinstance(self.deconvolved, bool):
            raise TypeError("Deconvolved must be a boolean.")
        self.estimated_parameters = {}
        self._projected = None
        self._history = None
        self._binary_images = {}
        self._log = {}

    def visualise(
        self, imtype: str='original', ind: int=1, save: bool=False, dest_dir: str=os.getcwd(),
        colmap: str='inferno', scale_bar: bool=True, bar_locate:str='upper left'
        ): 
        """ Visualise an image or stack slice. Plot saved as `{.title}_{._history}.png`.
        ...

        Arguments 
        ---------
        imtype : str='original'
            A string specifying whether to display `original` or `modified` data. 
        ind : int=1
            Index of slice to visualise from z-stack (first frame is n=1, also default value).
        save : bool=False
            Save plot to dest_dir. Displayed but not saved by default.
        dest_dir : str=os.getcwd()
            Directory to save plot in (defaults to current working directory) 
        colmap : str='inferno'
            Change color map (passed to `cmap` argument) in matplotlib.pyplot. Perceptually uniform colour map used by default.
        scale_bar : bool=True
            Adding a scale bar to images by default (provided, resolution is available).
        bar_locate : str='upper_left'
            Position of scale bar; upper left by default. 

        Returns
        -------
        matplotlib.pyplot
            Plot of specified slice from parent stack. 
        """
        if not isinstance(imtype, str) or not isinstance(ind, int):
            raise TypeError('imtype must be a string and ind must be an integer.')
        if scale_bar and self.resolution is None:
            raise ValueError('Resolution has not been specified.')
        if not isinstance(save, bool):
            raise TypeError('save must be a boolean.')
        if not os.path.exists(dest_dir):
            raise ValueError(f'Directory not recognised: {dest_dir}.')
        if imtype=='manipulated':
            if not self._history:
                raise ValueError('Raw data has not been processed yet.')
            if ind < 1 or ind > self.manipulated_depth:
                raise ValueError(f'ind must be an integer in range (1, {self.manipulated_depth})')        
            if self._projected or self.manipulated_depth == 1:
                plt.imshow(self.manipulated_stack, cmap=colmap)
            else: 
                plt.imshow(self.manipulated_stack[ind-1], cmap=colmap)        
        elif imtype=='original':
            if ind < 1 or ind > self.depth:
                raise ValueError(f'ind must be an integer in range (1, {self.depth})')  
            if self.depth > 1:
                plt.imshow(self.image_stack[ind-1], cmap=colmap)
            else: 
                plt.imshow(self.image_stack, cmap=colmap)
        else: 
            raise ValueError('Image type \'{imtype}\' not recognised; imtype must be one of [\'original\', \'manipulated\']'.format(imtype=imtype))
        if scale_bar:
            # convert barsize to nm if necessary because scale is specified in nm  
            barsize = self.resolution['pixel_size_xy']*1e3 if self.resolution['unit'] == 'micron' else self.resolution['pixel_size_xy']
            scalebar = ScaleBar(barsize, 'nm', box_color='None', color='#F2F2F2', location=bar_locate) 
            plt.gca().add_artist(scalebar)
        plt.axis('off')
        if save: 
            imtitle = f'{self.title.split(".")[0]}_{imtype}_{"_".join(self._history)}.png' if self._history else f'{self.title.split(".")[0]}_{imtype}.png'
            dest = os.path.join(dest_dir, imtitle)
            plt.savefig(dest, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show();


    def visualise_stack(
        self, imtype: str='original', substack=None, save: bool=False, dest_dir : str=os.getcwd(), 
        fig_size: tuple=None, colmap: str='inferno', scale_bar: bool=True, bar_locate:str='upper left'
        ):
        """ Visualise a stack as a tile of all constituent images. By default, all frames will be visualised. 
        ...
        Arguments 
        ---------
        imtype : str='original'
            A string specifying whether to display `original` or `modified` data. 
        substack : list
            A list of length=2, specifying the range [start_index, finish_index] of slices to perform the operation on.
            Note: indexing is from 1 to `n_frames` in image stack. 
        save : bool=False
            Save plot to dest_dir. Displayed but not saved by default.
        dest_dir : str=os.getcwd()
            Directory to save plot in (defaults to current working directory) 
        colmap : str='inferno'
            Change color map (passed to `cmap` argument) in matplotlib.pyplot. Perceptually uniform colour map used by default.
        scale_bar : bool=True
            Adding a scale bar to images by default (provided, resolution is available).
        bar_locate : str='upper_left'
            Position of scale bar; upper left by default. 

        Returns
        -------
        matplotlib.pyplot
            A tiled plot of specified substack from parent stack. 
        """
        if not isinstance(imtype, str):
            raise TypeError('imtype must be a string.')
        if substack and (len(substack) != 2 or not isinstance(substack, list)): 
            raise ValueError('substack has to be a list of length=2, specifying a range.')
        if not isinstance(save, bool):
            raise TypeError('save must be a boolean.')
        if not os.path.exists(dest_dir):
            raise ValueError(f'Directory not recognised: {dest_dir}.')
        if scale_bar and self.resolution is None:
            raise ValueError('Resolution has not been specified.')

        if imtype=='original':
            if substack is not None and (substack[0] < 1 or substack[1] > self.depth):
                raise ValueError(f'substack must be a list of integers in range (1, {self.depth}).')
            if substack: 
                data = self.image_stack[substack[0]-1:substack[1]].copy() 
                figrows, figcols = get_fig_dims(substack[1]-substack[0]+1)
            else:
                data = self.image_stack.copy()
                figrows, figcols = get_fig_dims(self.depth) 
                substack = [1, self.depth]
                
            absmin, absmax = np.min(data), np.max(data)
            plt.figure(figsize=(8, 8*(figrows/figcols))) 
            for n, image in enumerate(data[::1]):
                ax = plt.subplot(figrows,figcols,n+1)
                ax.imshow(image, cmap=colmap, vmin=absmin, vmax=absmax)
                if scale_bar:
                    # convert barsize to nm if necessary because scale is specified in nm  
                    barsize = self.resolution['pixel_size_xy']*1e3 if self.resolution['unit'] == 'micron' else self.resolution['pixel_size_xy']
                    scalebar = ScaleBar(barsize, 'nm', box_color='None', color='#F2F2F2', location=bar_locate) 
                    ax.add_artist(scalebar)
                ax.set_axis_off()
                ax.text(0.66, 0.05, 'n={count}'.format(count=substack[0]+n),color='#F2F2F2',transform=ax.transAxes,fontsize=8)

        elif imtype=='manipulated':
            if not self._history:
                raise ValueError('Raw data has not been processed yet.')
            if self._projected:
                self.visualise(imtype='manipulated',save=save,dest_dir=dest_dir,colmap=colmap,scale_bar=scale_bar,bar_locate=bar_locate)
                return None
            else:
                if substack is not None and (substack[0] < 1 or substack[1] > self.manipulated_depth):
                    raise ValueError(f'substack must be a list of integers in range (1, {self.manipulated_depth}).')
                if substack: 
                    data = self.manipulated_stack[substack[0]-1:substack[1]].copy() 
                    figrows, figcols = get_fig_dims(substack[1]-substack[0]+1)
                else:
                    data = self.manipulated_stack.copy()
                    manipulated_depth = data.shape[0]
                    figrows, figcols = get_fig_dims(manipulated_depth)
                    substack = [1, self.manipulated_depth]

                absmin, absmax = np.min(data), np.max(data)
                for n, image in enumerate(data[::1]):
                    ax = plt.subplot(figrows,figcols,n+1)
                    ax.imshow(image, cmap=colmap, vmin=absmin, vmax=absmax)
                    if (scale_bar and self.resolution is not None):
                        scalebar = ScaleBar(self.resolution, 'nm', box_color='None', color='#F2F2F2', location=bar_locate) 
                        ax.add_artist(scalebar)
                    ax.set_axis_off()
                    if substack and self.manipulated_substack_inds:
                        ax.text(0.66, 0.05, 'n={count}'.format(count=self.manipulated_substack_inds[0]-substack[0]+1+n),color='#F2F2F2',transform=ax.transAxes,fontsize=8)
                    elif self.manipulated_substack_inds and not substack:
                        ax.text(0.66, 0.05, 'n={count}'.format(count=self.manipulated_substack_inds[0]+n),color='#F2F2F2',transform=ax.transAxes,fontsize=8)
                    elif substack and not self.manipulated_substack_inds:
                        ax.text(0.66, 0.05, 'n={count}'.format(count=substack[0]+n),color='#F2F2F2',transform=ax.transAxes,fontsize=8)

        else:
            raise ValueError('Image type \'{imtype}\' not recognised; imtype must be one of [\'original\', \'manipulated\']'.format(imtype=imtype))

        plt.subplots_adjust(wspace=0.01, hspace=0.02)
        plt.tight_layout()

        if save==True: 
            imtitle = f'{self.title.split(".")[0]}_{imtype}_{"_".join(self._history)}.png' if self._history else f'{self.title.split(".")[0]}_{imtype}.png'
            dest = os.path.join(dest_dir, imtitle)
            plt.savefig(dest, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show();


    def normalise(self): 
        """ Normalises every frame in a z-stack (or just image) by minimum pixel intensity in that frame. Resulting frames have values in range [0,1].

        Returns
        -------
        self.manipulated_stack : np.ndarray
            Normalised image stack. 
        self.self.manipulated_depth : int
            Attribute matches self.depth.
        """
        if self.depth == 1: 
            self.manipulated_stack = (self.image_stack - np.min(self.image_stack)) / np.max(self.image_stack - np.min(self.image_stack))
        else:
            self.manipulated_stack =  np.array([(img - np.min(img)) / np.max(img - np.min(img)) for img in self.image_stack])
        self.manipulated_depth = np.copy(self.depth)
        self._call_hist('normalise')
        return None

 
    def z_project_min(self, substack=None):
        """ Returns the minimum axial projection of a z-stack. Retains the minimum intensity at every position.
        Can be performed on a sub-range.
        ...
        Arguments 
        ---------
        substack : list
            A list of length=2, specifying the range [start_index, finish_index] of slices to perform the operation on.
            Note: indexing is from 1 to `n_frames` in image stack. 
        Returns
        -------
        self.manipulated_stack : np.ndarray
            Minimum projection of dimensions=self.shape. 
        self.self.manipulated_depth : int
            Updated to 1.
        See also
        --------
        ActImg.z_project_max()
        """
        if not self._history or 'normalise' not in self._history: 
            use_raw = True
            #warnings.warn('Raw data has not been normalised; using raw data.') #raise ValueError
        else: 
            use_raw = False
        if substack and (len(substack) > 2 or not isinstance(substack, list)): 
            raise ValueError('substack has to be a list of length=2, specifying a range.')
        compare_depth = self.manipulated_depth if self.manipulated_depth > 0 else self.depth
        if substack and len(substack)==2 and (substack[0] < 1 or substack[1] > compare_depth):
            raise ValueError(f'substack must be a list of integers in range (1, {self.manipulated_depth}).')
        if substack and len(substack)==1:
            print('Cannot perform minimum projection on one frame, returning object.')
            return None
        else: 
            if substack and use_raw:
                data = self.image_stack[substack[0]-1:substack[1]].copy() 
            elif use_raw:
                data = self.image_stack.copy() 
            elif substack: 
                data = self.manipulated_stack[substack[0]-1:substack[1]].copy()
            else:
                data = self.manipulated_stack.copy()

            self.manipulated_stack = np.min(data,0)
            self.manipulated_depth = 1
            self._projected = True
            self._call_hist('z_project_min')
            return None


    def z_project_max(self, substack=None):
        """ Returns the maximum axial projection of a z-stack. Retains the maximum intensity at every pixel position. 
        Can be performed on a sub-range.
        ...
        Arguments 
        ---------
        substack : list
            A list of length=2, specifying the range [start_index, finish_index] of slices to perform the operation on.
            Note: indexing is from 1 to `n_frames` in image stack. 
        Returns
        -------
        self.manipulated_stack : np.ndarray
            Minimum projection of dimensions=self.shape. 
        self.self.manipulated_depth : int
            Updated to 1.
        See also
        --------
        ActImg.z_project_min()   
        """
        if not self._history or 'normalise' not in self._history: 
            use_raw = True
            #warnings.warn('Raw data has not been normalised; using raw data.') #raise ValueError
        else: 
            use_raw = False
        if substack and (len(substack) > 2 or not isinstance(substack, list)): 
            raise ValueError('substack has to be a list of length=2, specifying a range.')
        compare_depth = self.manipulated_depth if self.manipulated_depth > 0 else self.depth
        if substack and len(substack)==2 and (substack[0] < 1 or substack[1] > compare_depth):
            raise ValueError(f'substack must be a list of integers in range (1, {self.manipulated_depth}).')

        if substack and use_raw:
            if len(substack)==2:
                data = self.image_stack[substack[0]-1:substack[1]].copy() 
            elif len(substack)==1:
                data = self.image_stack[substack].copy() 
        elif use_raw and not substack:
            data = self.image_stack.copy() 
        elif substack and not use_raw: 
            if len(substack)==2:
                data = self.manipulated_stack[substack[0]-1:substack[1]].copy() 
            elif len(substack)==1:
                data = self.manipulated_stack[substack].copy() 
        else:
            data = self.manipulated_stack.copy()

        self.manipulated_stack = np.max(data,0) if len(data.shape)==3 else data
        self.manipulated_depth = 1
        self._projected = True
        self._call_hist('z_project_max')
        return None


    def threshold(self, threshold: float):
        """ Returns a binary thresholded image.
        Note: can be called after normalisation and  
        ...
        Arguments
        ---------
        threshold : float
            Values below the threshold are set to 1; the rest are set to 0. 
        Returns
        -------
        self.manipulated_stack : np.ndarray of ints 
            Binary image of dimensions=self.shape. 
        self.self.manipulated_depth : int
            Updated to 1.
        """
        if not self._history or 'normalise' not in self._history: 
            raise ValueError('Raw data has not been normalised.')
        if self._projected is None: 
            print('Data has not been projected in z.') #raise ValueError
        if not isinstance(threshold, float):
            raise TypeError('Threshold must be a float.')
        if threshold >=1: 
            raise ValueError('Threshold cannot be >= 1.')

        self.manipulated_stack = (self.manipulated_stack < threshold).astype('int')
        self.manipulated_depth = 1
        self._call_hist('threshold')
        return None


    def threshold_dynamic(self, std_dev_factor: float=0.0, return_mean_std_dev: bool=False, line_prof_coords=None): 
        """ Obtains line profiles (of width 5 pixels) specified in boundaries `line_prof_coords`, returns mean and standard deviation of aggregated profile.
        
        Arguments
        ---------
        std_dev_dactor : float=0.0
            ???
        return_mean_std_dev : bool=False
            ??? 
        line_prof_coords : nested list of tuples 
            List of tuples [[start, end], ...] where start = (start_row, start_col) and end = (end_row, end_col) specify the start and end point of the profile, respectively. 
        Returns
        -------
        self.manipulated_stack : np.ndarray of ints 
            Binary image of dimensions=self.shape. 
        self.self.manipulated_depth : int
            Updated to 1.
        (mean, std_dev) : tuple of floats (optional) 
            Optionally return mean and standard deviation of the aggregated line profiles. 
        See also
        --------
        ActImg._threshold_preview_cases()
        """
        if not self._history or 'normalise' not in self._history: 
            raise ValueError('Raw data has not been normalised.')
        if self._projected is None: 
            print('Data has not been projected in z.') #raise ValueError
        if not isinstance(std_dev_factor, (float, int)): 
            raise TypeError('`std_dev_factor` must be float or int.') 
        if std_dev_factor < 0: 
            raise ValueError('`std_dev_factor` must be >=0.') 
        if not isinstance(return_mean_std_dev, bool):
            raise TypeError('`return_mean_st_dev` must be a boolean.')

        self._aggregate_line_profiles(line_prof_coords=None)
        mean, std_dev = self.estimated_parameters['aggregated_line_profiles'].values()

        # threshold is 'inverse' i.e. 1 if value < threshold else 0    
        self.manipulated_stack = (self.manipulated_stack < mean-std_dev_factor*std_dev).astype('int')
        self._call_hist('threshold')
        self.manipulated_depth = 1


        if return_mean_std_dev:
            return mean, std_dev 
        else:
            return None
            
    

    def _aggregate_line_profiles(self, line_prof_coords=None, return_mean_st_dev: bool=False):
        # default profiles 
        if not line_prof_coords: 
            rows, cols = self.shape
            line_prof_coords = [[(0,0),(rows,cols)], [(rows,0),(0,cols)],
                                [(int(rows/2),0),(int(rows/2),cols)], [(0,int(cols/2)),(rows,int(cols/2))]]
        line_profs = [None]*len(line_prof_coords) 

        try:
            for n, (start, end) in enumerate(line_prof_coords): 
                line_profs[n] = profile_line(self.manipulated_stack, start, end, linewidth=5).ravel()

            all_profs = np.concatenate(line_profs).ravel() # shape (n,)
            mean, std_dev = np.mean(all_profs), np.std(all_profs)
            self.estimated_parameters['aggregated_line_profiles'] = {'mean': mean, 'std_dev': std_dev} 

            if return_mean_st_dev:
                return mean, std_dev 
            else:
                return None
        except: 
            raise RuntimeError('line_prof_coords cannot be unpacked to extract line profiles.')


    def _threshold_preview_cases(self, mean: float=None, std_dev: float=None, factors=None, max_proj_substack=None, save: bool=False, dest_dir: str=os.getcwd()):
        """ Helper previews the outputs of several degrees of thresholding mean-factor*std_dev. 
        Note: must be applied after steerable filter. 

        Arguments
        ---------
        mean : float, optional
            Mean of line profiles, optionally returned by ActImg.threshold_dynamic()
        std_dev : float, optional
            Standard deviation of line profiles, optionally returned by ActImg.threshold_dynamic()
        factors : list of floats or ints
            The factors where threshold = mean-factor*std_dev for factor in factors
        save : bool=False
            Plot is displayed but not saved by default. True displays and saves plot. 
        dest_dir : str=os.getcwd()
            Destination where the plot is saved if save=True; defaults to current working directory. 

        Returns
        -------
        matplotlib.pyplot
            Plots a tile comparing the minimum projection of oriented filter responses to the different thresholds.

        See also
        --------
        ActImg.threshold_dynamic()
        """
        if (mean and not isinstance(mean, float)) or (std_dev and not isinstance(std_dev, float)):
            raise TypeError('Mean and std_dev must both be floats.')
        if mean is None and std_dev is None:
            try:
                mean, std_dev = self.estimated_parameters['aggregated_line_profiles'].values()
            except KeyError:
                self._aggregate_line_profiles(line_prof_coords=None)
                mean, std_dev = self.estimated_parameters['aggregated_line_profiles'].values()
                #raise AttributeError('ActImg.estimated_parameters mean and std_dev (key=`aggregated_line_profiles`) not found, please specify (mean, std_dev) or call `ActImg.threshold_dynamic()`.')
            finally: 
                pass
        try: 
            factors = np.array(factors)
        except: 
            raise TypeError('Factor must be array-like.')

        img = self.manipulated_stack.copy()
        # create a copy of instance for maximum projection 
        tmp_actimg = ActImg(self.image_stack, title='tmp',shape=self.shape, depth=self.depth, deconvolved=True,
                              resolution=self.resolution, meta=self.meta)
        tmp_actimg.normalise()
        tmp_actimg.z_project_max(max_proj_substack)
        img_max_proj = tmp_actimg.manipulated_stack.copy()

        threshold_variants = ['img']
        threshold_variants.extend([f'img < mean-{factor}*std_dev' for factor in factors])
        plt_titles = ['img'] + [f'img < $\\mu$-{factor:.2f}*$\\sigma$' for factor in factors]


        figrows, figcols = get_fig_dims(len(threshold_variants))
        for n, (exp, title) in enumerate(zip(threshold_variants, plt_titles)):
            ax = plt.subplot(figrows,figcols,n+1)
            if 'mean' in exp:
                thresh = eval(exp)
                ax.imshow(thresh, cmap='gray')
                ax.imshow(img_max_proj, cmap='gray', alpha=0.7)
                ax.set_title(title)
            else: 
                ax.imshow(eval(exp), cmap='gray')
                ax.set_title('Response')
            ax.set_axis_off()
        plt.subplots_adjust(wspace=0.02, hspace=0.2)
        plt.tight_layout()
        plt.suptitle(f'Threshold for $\mu$ = {mean:.5f} and $\sigma$ = {std_dev:.5f}')
        
        if save: 
            imtitle = f'{self.title.split(".")[0]}_mean_{mean:.5f}_std_dev_{std_dev:.5f}.png'
            dest = os.path.join(dest_dir, imtitle)
            plt.savefig(dest, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
        else: 
            plt.show();


    def steerable_gauss_2order(self, substack=None, sigma: float=2.0, theta: float=0., visualise: bool=True, tmp: bool=False):
        """ Steers an X-Y separable second order Gaussian filter in direction theta.
        Implemented according to W. T. Freeman and E. H. Adelson, "The Design and Use of Steerable Filters", IEEE PAMI, 1991.
        Based on matlab code from Jincheng Pang, Tufts University, 2013.
        ...
        Arguments 
        ---------
        substack : list
            A list of length=2, specifying the range [start_index, finish_index] of slices to perform the operation on.
            Note: indexing is from 1 to `n_frames` in image stack. 
        sigma : float=2.0
            The standard deviation of the second-order Gaussian kernel.
        theta : float=0.
            The steerable filter orientation, in degrees. 
        tmp : bool=False
            Optional argument enables returning the response and the oriented filter without updating original object. 

        Returns
        -------
        self.manipulated_stack : np.ndarray 
            An n-dimensional array of oriented filter responses for ever . 
        self.self.manipulated_depth : int
            Updated to according to number of slices processed (specified by `substack`).
        self.manipulated_substack_inds : list=substack
            Substack inds are updated to yield correct visualisation using ActImg.visualise_stack() method
        dictionary (optional)
            Dictionary maps the response of the theta-rotated derivative and the oriented filter. 
        """
        if not self._history or 'normalise' not in self._history: 
            raise ValueError('Raw data has not been normalised.') #use_raw = True
        if substack and (len(substack) > 2 or not isinstance(substack, list)): 
            raise ValueError('substack has to be a list of length 1 or 2, specifying a range.')
        
        if substack and len(substack)==1:
            depth, rows, cols = 1, *self.shape
            data = np.empty((depth, rows, cols))
            data[0] = self.manipulated_stack[substack[0]-1].copy()
        elif substack and len(substack)==2: 
            if substack is not None and (substack[0] < 1 or substack[1] > self.manipulated_depth):
                raise ValueError(f'substack must be a list of integers in range (1, {self.manipulated_depth}).')
            data = self.manipulated_stack[substack[0]-1:substack[1]].copy()
            depth, rows, cols = substack[1]-substack[0]+1, *self.shape
        elif substack is None:
            data = self.manipulated_stack.copy()
            depth, rows, cols = self.manipulated_depth, *self.shape
        else: 
            print('Problem loading the data')

        #### Separable filter kernels
        # Gaussian kernel mesh grid  
        Wx = np.floor((8/2)*sigma)
        Wx = Wx if Wx >= 1 else 1

        x = np.arange(-Wx,Wx+1) # determines kernel size 
        xx,yy = np.meshgrid(x,x)

        theta = np.deg2rad(-theta) # convert to radians, clockwise


        # Second derivative of the Gaussian
        g0 = np.array(np.exp(-(xx**2+yy**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)))
        # Separable filter kernels 
        G2a = np.array(-g0/sigma**2+g0*xx**2/sigma**4)
        G2b =  np.array(g0*xx*yy/sigma**4)
        G2c = np.array(-g0/sigma**2+g0*yy**2/sigma**4)

        oriented_filter = np.array((np.cos(theta))**2*G2a + np.sin(theta)**2*G2c - 2*np.cos(theta)*np.sin(theta)*G2b)

        response_stack = np.empty((depth, rows, cols))
        response_stack.fill(np.nan)

        if depth > 1:
            for count, input in enumerate(data): 
                #### Oriented filter response.
                # Calculate image gradients, using separability
                # I2a = correlate(input, G2a, mode='nearest')
                # I2b = correlate(input, G2b, mode='nearest')
                # I2c = correlate(input, G2c, mode='nearest')
                I2a = cv2.filter2D(input, -1, G2a, borderType=cv2.BORDER_REPLICATE)
                I2b = cv2.filter2D(input, -1, G2b, borderType=cv2.BORDER_REPLICATE)
                I2c = cv2.filter2D(input, -1, G2c, borderType=cv2.BORDER_REPLICATE)

                # Evaluate oriented filter response.
                response = np.array(
                    (np.cos(theta))**2*I2a + np.sin(theta)**2*I2c - 2*np.cos(theta)*np.sin(theta)*I2b) #, dtype='uint16')
                response_stack[count,:,:] = np.copy(response)
        else:
            input = data[0,:,:]
            # input = data[:,:] # use for debugging
            # I2a = correlate(input, G2a, mode='nearest')
            # I2b = correlate(input, G2b, mode='nearest')
            # I2c = correlate(input, G2c, mode='nearest')
            I2a = cv2.filter2D(input, -1, G2a, borderType=cv2.BORDER_REPLICATE)
            I2b = cv2.filter2D(input, -1, G2b, borderType=cv2.BORDER_REPLICATE)
            I2c = cv2.filter2D(input, -1, G2c, borderType=cv2.BORDER_REPLICATE)

            # Evaluate oriented filter response.
            response_stack[0,:,:] = np.array(
                (np.cos(theta))**2*I2a + np.sin(theta)**2*I2c - 2*np.cos(theta)*np.sin(theta)*I2b) #, dtype='uint16')


        if visualise: 
            for n, (image, title) in enumerate(zip( 
                [*np.rollaxis(data, 0), *np.rollaxis(response_stack, 0)],# oriented_filter], 
                [*['Input']*depth, *['Response']*depth]# 'Oriented filter']
                )):
                ax = plt.subplot(2,depth,n+1)
                ax.imshow(image, cmap='gray')
                ax.set_axis_off()
                ax.set_title(title)
            plt.subplots_adjust(wspace=0.01, hspace=0.01)
            plt.tight_layout()
            plt.show();

        if tmp: 
            return {'response': response_stack, 'filter': oriented_filter}
        else:
            self.manipulated_stack = response_stack
            self.manipulated_depth = depth
            self.manipulated_substack_inds = substack
            self._call_hist('steerable_gauss_2order')
            return None


    def steerable_gauss_2order_thetas(self, thetas, sigma: float=2.0, substack=None, visualise=False, return_responses=False):
        """ Applies steerable second order Gaussian filters oriented in multiple directions (specified by `thetas`). For every frame, the minimum projection of the oriented filter responses is returned in an n-dimensional array. 
        ...
        Arguments
        ---------
        thetas : list
            A list of floats or integers, specifying the directions (in degrees) in which the second-order Gaussian should be steered.
        sigma : float=2.0
            The standard deviation of the Gaussian.
        substack : list
            A list of length=2, specifying the range [start_index, finish_index] of slices to perform the operation on.
            Note: indexing is from 1 to `n_frames` in image stack. 
        visualise : bool=False
            Optionally, visualise the response stack. 
        Returns 
        -------
        self.manipulated_stack : np.ndarray 
            An n-dimensional array (where n is determined by `substacks`), where each frame is the minimum projection of the oriented filter responses. 
        self.self.manipulated_depth : int
            Updated to according to number of slices processed (specified by `substack`).
        self.manipulated_substack_inds : list=substack
            Substack inds are updated to yield correct visualisation using ActImg.visualise_stack() method
        See also
        --------
        ActImg.steerable_gauss_2order()
        TODO: input validation, visualisation vmax/vmin
        """
        results = dict.fromkeys(thetas)
        for angle in thetas:
            results[angle] = self.steerable_gauss_2order(theta=angle, substack=substack,sigma=sigma,visualise=False,tmp=True)
        response_stack = np.array([value['response'] for _, value in results.items()])
        if return_responses: 
            responses_out = response_stack.copy()
        response_stack = np.min(response_stack, 0)

        if visualise:
            titles = [f'n_{str(n)}' for n in np.arange(substack[0],substack[1]+1)]
            figrows, figcols = get_fig_dims(len(titles))
            for n, (image, title) in enumerate(zip(np.rollaxis(response_stack, 0), titles)):
                ax = plt.subplot(figrows,figcols,n+1)
                ax.imshow(image, cmap='gray')
                ax.set_axis_off()
                ax.set_title(title)
            plt.subplots_adjust(wspace=0.02, hspace=0.02)
            plt.tight_layout()
            plt.show();


        self.manipulated_stack = response_stack
        if substack and len(substack)==1:
            self.manipulated_depth = 1
        elif substack and len(substack)==2: 
            self.manipulated_depth = substack[1]-substack[0]+1
        self.manipulated_substack_inds = substack
        theta_string = f'+{thetas[0]}+{thetas[-1]}+{len(thetas)}'
        self._call_hist('steerable_gauss_2order_thetas')#+theta_string)
        if return_responses: 
            return responses_out
        else:
            return None 


    def _get_oriented_filter(self, theta: float, sigma: float):
        """ Helper method; returns a second-order Gaussian filter with sigma=`sigma` oriented clockwise by angl=`theta`. 
        ...
        Arguments
        ---------
        theta : float=0.
            The steerable filter orientation, in degrees. 
        sigma : float=2.0
            The standard deviation of the second-order Gaussian kernel.
        Returns
        -------
        oriented_filter : np.ndarray
            The kernel of an oriented/steered second-order Gaussian. 
        See also
        --------
        ActImg.steerable_gauss_2order()
        ActImg._visualise_oriented_filters()
        TODO: input validation
        """
        #### Separable filter kernels
        # Gaussian kernel mesh grid  
        Wx = np.floor((8/2)*sigma)
        Wx = Wx if Wx >= 1 else 1
        x = np.arange(-Wx,Wx+1) # determines kernel size 
        xx,yy = np.meshgrid(x,x)
        theta = np.deg2rad(-theta) # convert to radians, clockwise
        # Second derivative of the Gaussian
        g0 = np.array(np.exp(-(xx**2+yy**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)))
        # Separable filter kernels 
        G2a = np.array(-g0/sigma**2+g0*xx**2/sigma**4)
        G2b =  np.array(g0*xx*yy/sigma**4)
        G2c = np.array(-g0/sigma**2+g0*yy**2/sigma**4)
        oriented_filter = np.array((np.cos(theta))**2*G2a + np.sin(theta)**2*G2c - 2*np.cos(theta)*np.sin(theta)*G2b)
        return oriented_filter 
    

    def _visualise_oriented_filters(self, thetas: list, sigma: float, save: bool=False, dest_dir: str=os.getcwd(), return_filters=False): 
        """ Helper method; tile of plots shows the oriented filters; optionally, the plot can be saved or the oriented filters can be returned as a dictionary.
        ...
        Arguments
        --------- 
        thetas : list
            A list of floats or integers, specifying the directions (in degrees) in which the second-order Gaussian should be steered.
        sigma : float=2.0
            The standard deviation of the Gaussian.
        save : bool=False
            Plot is displayed but not saved by default. True displays and saves plot. 
        dest_dir : str=os.getcwd()
            Destination where the plot is saved if save=True; defaults to current working directory. 
        return_filters : bool=False
            Optionally, return the oriented filter as a dictionary. 
        Returns
        -------
        matplotlib.pyplot
            Plot of specified slice from parent stack.
        png (optional)
            A png of the plot can be saved to the `dest_dir`. 
        all_filters_dict : dictionary (optional)
            A dictionary which maps all `thetas` to oriented filters.  
        See also
        --------
        ActImg.steerable_gauss_2order()
        ActImg._get_oriented_filter()
        TODO: input validation, visualisation vmax/vmin
        """
        all_filters = []
        for angle in thetas:
            all_filters.append(self._get_oriented_filter(theta=angle,sigma=sigma))
        titles = [f'$\\theta$ = {n:.2f}' for n in thetas]
        figrows, figcols = get_fig_dims(len(thetas)) 
        for n, (image, title) in enumerate(zip(np.rollaxis(np.asarray(all_filters), 0), titles)):
            ax = plt.subplot(figrows,figcols,n+1)
            ax.imshow(image, cmap='gray')
            ax.set_axis_off()
            ax.set_title(title)
        plt.subplots_adjust(wspace=0.01, hspace=0.03)
        plt.tight_layout()
        if save: 
            imtitle = f'{self.title.split(".")[0]}_oriented_filters{"+".join(str(i) for i in [thetas[0], thetas[-1], len(thetas)])}.png'
            dest = os.path.join(dest_dir, imtitle)
            plt.savefig(dest, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()
        plt.show();
        if return_filters:
            all_filters_dict = dict.fromkeys(thetas)
            for angle, filter in zip(thetas, all_filters):
                all_filters_dict[angle] = filter 
            return all_filters_dict
        else: 
            return None
            

    def surface_area(self, verbose=False, visualise=False):
        """ Accepts a binary image and returns the segmented cell surface area with associated units. 
        Recommended use with basal membrane manipulation.
        Note: the algorithm uses serial dilations to include the periphery of the cell even if the cell boundary is discontinuous.
        The dilations are followed by an equivalent number of serial erosions to avoid overestimating the cell area. 
        Note: it is assumed that the largest object in the field of view is the cell which is to be segmented.
        ...
        Arguments
        ---------
        verbose : bool=False
            Optionally, print out mesh density percentage and definition. 
        visualise : bool=False
            Optionally, visualise the segmented cell surface against the segmented mesh.  
        Returns
        ------
        self.estimated_parameters['surface_area'] : float
            The segmented cell surface area, defined as the sum of the labelled pixels * size of one pixel ($`pixel_size_xy`^2$).
        Raises
        ------
        ValueError
            If the manipulated_stack is not a binary array with values [0,1].
        """
        if not (np.sort(np.unique(self.manipulated_stack)) == [0,1]).all():
            raise ValueError('Input image is not binary.')

        img = self.manipulated_stack.copy()
        # close with small 2x2 structure (ensure connectivity) 
        closed_image = binary_closing(img, structure = np.ones((2,2))) 
        # dilate serially (ensure connectivity)
        dilated = ski_morph.dilation(closed_image).astype('int')
        dilated = ski_morph.dilation(dilated).astype('int')
        dilated = ski_morph.dilation(dilated).astype('int')
        # fill to segment prelim cell surface
        filled_image = morphology.binary_fill_holes(dilated)
        # find contour to get cell edges
        contours = measure.find_contours(filled_image)
        ind_max = np.argmax([len(x) for x in contours])
        cont_img = np.zeros(filled_image.shape)
        cont_img[np.ceil(contours[ind_max]).astype('int')[:,0], np.ceil(contours[ind_max]).astype('int')[:,1]] = 1
        # dilate contour to avoid discontinuity within self
        dilated_cont = ski_morph.dilation(cont_img).astype('int')
        # fill contour to segment final cell edges
        filled_cont = morphology.binary_fill_holes(dilated_cont).astype('int')
        # erode serially (equal to no. dilations, avoid overestimation)
        eroded_cont = ski_morph.binary_erosion(filled_cont).astype('int')
        eroded_cont = ski_morph.binary_erosion(eroded_cont).astype('int')
        eroded_cont = ski_morph.binary_erosion(eroded_cont).astype('int')

        if visualise:
            plt.imshow(img)
            plt.imshow(eroded_cont, cmap='gray', alpha=0.5)
            plt.show()

        surface_area, unit = np.sum(eroded_cont)*(self.resolution['pixel_size_xy']**2), self.resolution['unit']

        if surface_area < np.sum(img)*(self.resolution['pixel_size_xy']**2) - surface_area:
            self._update_log('failed_segmentation', {'imtitle': self.title, 'parameter': 'surface_area'})
            if verbose: 
                print(f'Image not segmented well: {self.title}')
        else: 
            if (self.resolution['pixel_size_xy'] == 'micron') | (surface_area > 1e4):
                surface_area, unit = surface_area/1e6, 'um'

            self.estimated_parameters['surface_area'] = {'area': surface_area, 'unit': unit}
            self._call_hist('surface_area')
            self._binary_images['cell_surface'] = eroded_cont.copy()

            if verbose: 
                print(f'Segmented cell surface area ({self.resolution["unit"]}^2) is {surface_area:.0f}.')
                print('Defined as the difference between the filled and unfilled mask.')
        return None

    def meshwork_density(self, verbose=False):
        """ Accepts a binary image and returns the meshwork density (percentage).
        ...
        Arguments
        ---------
        verbose : bool=False
            Optionally, print out mesh density percentage and definition. 
        Returns
        ------
        self.estimated_parameters['mesh_density_percentage'] : float
            A percentage of mesh density, calculated as the difference between unfilled and filled image over the sum of the filled image. 
        Raises
        ------
        ValueError
            If the manipulated_stack is not a binary array with values [0,1].
        """
        if not (np.sort(np.unique(self.manipulated_stack)) == [0,1]).all():
            raise ValueError('Input image is not binary.')
        
        img = self.manipulated_stack.copy()
        try:
            cell_coords = np.nonzero(self._binary_images['cell_surface'])
        except: 
            self.surface_area()
            cell_coords = np.nonzero(self._binary_images['cell_surface'])
        finally: 
            # redefine density as difference between filled and unfilled surface, discard rest  
            # close any very small holes to ensure uniformity 
            closed_img = morphology.binary_closing(img, structure=np.ones((2,2)))
            # fill any holes in closed image
            filled_img = morphology.binary_fill_holes(closed_img)

            mesh_density = np.sum(img)*100 / np.sum(filled_img)
            if verbose: 
                print(f'The percentage mesh density is  {mesh_density:.2f} %')
                print('Defined as the difference between the filled and unfilled mask.')

            self.estimated_parameters['mesh_density_percentage'] = mesh_density
            self._call_hist('meshwork_density')
            return None


    def meshwork_size(self, summary: bool=False, verbose: bool=False, visualise=False, save_vis=False, dest_dir: str=os.getcwd()):
        """ Accepts a binary image and returns meshwork size. 
        Uses skimage.measure.regionprops - `equivalent_diameter_area`
        'The diameter of a circle with the same area as the region.'
        Arguments
        ---------
        summary : bool=False
            Optionally, calculate 
        verbose : bool=False
            Optionally, print out summary statistics for mesh size. Must have `summary=True` to print.
        visualise : bool=False
            Optionally, visualise the binary, filled, inverted, and labelled images. 
        save_vis : bool=False
            Optionally, save a plot with the results of the filling and labelling of the binary image. 
        Returns
        -------
        self.estimated_parameters['equivalent_diameters'] : pandas DataFrame
            A numpy.ndarray containing all label equivalent_diameter_area properties from `measure.regionprops()`. 
            The default unit of the equivalent diameters will be the unit in self.resolution['unit'], which defaults to nm.
        self.estimated_parameters['mesh_size_summary'] : dict, optional
            A dictionary mapping the mean, 95% CI of the mean, median, and IQR of the `equivalent_diameters` above.
        Raises
        ------
        ValueError
            If the manipulated_stack is not a binary array with values [0,1]. 
        """
        if not (np.sort(np.unique(self.manipulated_stack)) == [0,1]).all():
            raise ValueError('Input image is not binary.')
        if not all(list(map(lambda x: isinstance(x, bool), [summary, verbose, save_vis]))):
            raise TypeError('`summary`, `verbose`, `save_vis` must all be boolean.')
        if not os.path.exists(dest_dir):
            raise FileExistsError(f'Path not found: {dest_dir}')
        
        img = self.manipulated_stack.copy()
        img_inverted = (img==0).astype('int')
        try: 
            self._binary_images['cell_surface'] is not None
        except KeyError:
            self.surface_area()
            self._binary_images['cell_surface']
        finally: 
            labels = measure.label(img_inverted)
            equiv_diams = pd.DataFrame(measure.regionprops_table(labels, img, properties=['equivalent_diameter_area']))*self.resolution['pixel_size_xy']
            self.estimated_parameters['equivalent_diameters'] = {'values': equiv_diams.iloc[:,0].values.tolist(), 'unit': self.resolution['unit']} 
            self._binary_images['inverted_image'] = img_inverted.copy()
            self._call_hist('meshwork_size')

            if summary: 
                median = equiv_diams.iloc[:,0].median()
                mean = equiv_diams.iloc[:,0].mean()
                CI95_mean_norm_lower, CI95_mean_norm_upper = stats.norm.interval(alpha=0.95, loc=mean, scale=equiv_diams.iloc[:,0].sem())
                Q25, Q75 = equiv_diams.iloc[:,0].quantile([0.25, 0.75])
                self.estimated_parameters['mesh_size_summary'] = {'mean': mean, '95%_CI': [CI95_mean_norm_lower, CI95_mean_norm_upper],
                                                                'median': median, 'IQR': [Q25, Q75]}
            if verbose and summary: 
                print(f'mean mesh size ({self.resolution["unit"]}):        {mean:.3f}')
                print(f'95% CI of the mean:    [{CI95_mean_norm_lower:.3f}, {CI95_mean_norm_upper:.3f}]')
                print(f'median mesh size ({self.resolution["unit"]}):      {median:.3f}')
                print(f'IQR of mesh sizes:     [{Q25:.3f}, {Q75:.3f}]')
            
            if visualise or save_vis: 
                plt.subplot(2,2,1)
                plt.imshow(img, cmap='gray')
                plt.title('binary image')
                plt.axis('off')
                plt.subplot(2,2,2)
                plt.imshow(self._binary_images['cell_surface'], cmap='gray')
                plt.title('cell surface')
                plt.axis('off')
                plt.subplot(2,2,3)
                plt.imshow(img_inverted, cmap='gray')
                plt.title('inverted image')
                plt.axis('off')
                plt.subplot(2,2,4)
                plt.imshow(labels, cmap='tab20c_r')
                plt.title('inverted image labels')
                plt.axis('off')
                plt.tight_layout()

                if save_vis: 
                    imtitle = f'{self.title.split(".")[0]}_mesh_segmentation.png'
                    dest = os.path.join(dest_dir, imtitle)
                    plt.savefig(dest, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
                    plt.close()
                else: 
                    plt.show()
            return None


    def nuke(self):
        """ Restores an ActImg instance to it's pre-manipulated state.
        Returns
        -------
        self : ActImg
            Returns the same instance of ActImg but without any manipulation and history. 
        """
        self.manipulated_stack = None
        self.manipulated_depth = 0
        self.manipulated_substack_inds = None
        self._projected = None
        self._history = None
        return None
    

    def save(self, dest_dir: str=os.getcwd()):
        """ Save ActImg object using pickle. 
        Arguments
        ---------
        dest_dir : str or Path
            Path where the ActImg object should be saved in `.pkl` format; defaults to current working directory.   
        Returns
        -------
            A pickle file (`.pkl`) with original image title in filename. 
        """
        if not os.path.exists(dest_dir):
            raise FileNotFoundError(f'Path not found: {dest_dir}')
        
        title = 'actimg_'+self.title.split(".")[0]+'.pkl' 
        with open(os.path.join(dest_dir, title), 'wb') as f:
            pickle.dump(self, f, -1)
        return None


    def save_estimated_params(self, dest_dir: str=os.getcwd()):
        """ Saves estimated parameters in JSON format.
        Arguments
        ---------
        dest_dir : str or Path
            Path of destination where the JSON file should be saved; defaults to current working directory.  
        Rreturns
        --------
            A JSON file with the `estimated parameters` in the ActImg object and the resolution-related parameters from `self.resolution`.
        """
        json_obj = json.dumps({'estimated_parameters': self.estimated_parameters,
                               'resolution': self.resolution})
        file = 'actimg_params_'+self.title.split(".")[0]+'.json' 
        with open(os.path.join(dest_dir, file), 'w') as f:
           f.write(json_obj)
        return None



    def _call_hist(self, action):
        """ Helper records performed manipulations (called methods) in a list.
        Returns
        -------
        self._history : list of str
            A list of strings which specify the processing steps applied (firstt to latest).  
        """
        if self._history is None: 
            self._history = []
        self._history.append(action)
        return None
    
    def _update_log(self, key, value):
        """ Helper updates the `self._log` attribute with any pertinent information. """
        try: 
            self._log[key].append(value)
        except KeyError: 
            self._log[key] = value



def get_ActImg(image_name: str, image_dir: str):
    """ Creates an ActImg instance. 
    Arguments
    ---------
    image_name : str
        A string specifies the name of the image contained in `image_dir`. Used for instance `title` attribute.
    image_dir : str
        The root directory where `image_name` is contained. 
    Returns
    -------
    ActImg
        An instance of the ActImg class.   
    """
    img, title = get_image_stack(os.path.join(image_dir, image_name), verbose=False)
    shape = img[0].shape
    depth = len(img)
    if depth == 1:      # extract from nested tuple
        img = img[0]
    devonv = True if 'deconv' in image_name else False
    metadata = get_meta(os.path.join(image_dir, image_name))
    resolut = get_resolution(metadata.copy(), nm=True)
    return ActImg(image_stack=np.asarray(img), title=title, shape=shape, depth=depth, deconvolved=devonv, 
                  resolution=resolut, meta=metadata)


def load_ActImg(obj_path: str):
    """ Imports a saved ActImg object. 
    Arguments
    ---------
    obj_path : str or Path
        Path where ActImg object is stored in pickle format (must have the `.pkl` extension.) 
    Returns
    -------
    actimg : ActImg object 
        An instance of the ActImg class. 
    """
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f'File not found: `{obj_path}`')
    if not obj_path.endswith('.pkl'): 
        raise TypeError(f'File extension not recognised: {obj_path.split(".")[-1]}')
    
    with open(obj_path, 'rb') as f:
        actimg = pickle.load(f)
    return actimg 
