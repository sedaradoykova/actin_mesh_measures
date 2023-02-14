import os, cv2, warnings
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import correlate, morphology
from skimage import measure
from dataclasses import dataclass
from actin_meshwork_analysis.meshwork.utils import get_image_stack, get_meta, get_resolution, get_fig_dims
 
""" Design considerations: 
    Currently, the original data is kept and modifications happen on a copy which is internally updated 
    after methods are called on it with the option of nuking the modifications to restore original object. 
    A history of all changes implemented (methods called) are recorded in a hidden attribute ._history. 

    TODO: 
        - [ ] Annotate steps of analysis in docstrings? or separately
        - [ ] meshwork_size and meshwork_density methods
        - [ ] add voxel size attribute (i.e. resolution in z) in metadata  
        - [ ] test steerable second order filter but how?
        - [ ! ] steerable filter: results inconsistent with matlab (see tests/test_steer_gauss*) 
        - when to return self?   
"""

@dataclass()
class ActinImg:
    image_stack: tuple or np.ndarray
    title: str
    shape: tuple=None
    depth: int=None
    deconvolved: bool=None
    resolution: float=None
    res_units: str=''
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
        self._projected = None
        self._history = None


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
            Change color map (passed to cmap argument) in matplotlib.pyplot.
        scale_bar : bool=True
            Adding a scale bar to images by default (provided, resolution is available).
        bar_locate : str='upper_left'
            Position of scale bar; upper left by default. 
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
            scalebar = ScaleBar(self.resolution, 'nm', box_color='None', color='#F2F2F2', location=bar_locate) 
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
            Change color map (passed to cmap argument) in matplotlib.pyplot.
        scale_bar : bool=True
            Adding a scale bar to images by default (provided, resolution is available).
        bar_locate : str='upper_left'
            Position of scale bar; upper left by default. 
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
                    scalebar = ScaleBar(self.resolution, 'nm', box_color='None', color='#F2F2F2', location=bar_locate) 
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
        """ Normalises every frame in a z-stack (or just image) by minimum pixel intensity in that frame. 
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
        """
        if not self._history or 'normalise' not in self._history: 
            raise ValueError('Raw data has not been normalised.')
        if self._projected is None: 
            print('Data has not been projected in z.') #raise ValueError
        if not isinstance(threshold, float):
            raise TypeError('Threshold must be a float.')
        if threshold < 0 or threshold >=1: 
            raise ValueError('Threshold cannot be <0 or >= 1.')

        self.manipulated_stack = (self.manipulated_stack > threshold).astype('int')
        self._call_hist('threshold')
        return None

        
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
            The standard deviation of the Gaussian.
        theta : float=0.
            The steerable filter orientation. 
        tmp : bool=False
            Optional argument enables returning the response and the oriented filter without updating original object. 

        Returns
        -------
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
        else:
            data = self.manipulated_stack.copy()
            depth, rows, cols = self.manipulated_depth, *self.shape

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


    def steerable_gauss_2order_thetas(self, thetas, sigma: float=2.0, substack=None, visualise=False):
        """ Applies steerable second order Gaussian filters oriented in multiple directions (specified by `thetas`). 
        ...
        Arguments
        ---------
        thetas : list
            A list of floats or integers, specifying the directions in which the Gaussian should be steered.
        sigma : float=2.0
            The standard deviation of the Gaussian.
        substack : list
            A list of length=2, specifying the range [start_index, finish_index] of slices to perform the operation on.
            Note: indexing is from 1 to `n_frames` in image stack. 
        Returns 
        -------
        TODO: finish docstrings, input validation, visualisation vmax/vmin
        """
        results = dict.fromkeys(thetas)
        for angle in thetas:
            results[angle] = self.steerable_gauss_2order(theta=angle, substack=substack,sigma=sigma,visualise=False,tmp=True)
        response_stack = np.array([value['response'] for key, value in results.items()])
        response_stack = np.mean(response_stack, 0)

        if visualise:
            titles = [f'n_{str(n)}' for n in np.arange(substack[0],substack[1]+1)]
            figrows, figcols = get_fig_dims(len(thetas))
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
        self._call_hist('steerable_gauss_2order_thetas'+theta_string)
        return None 

    def _get_oriented_filter(self, theta, sigma):
        """ Helper method; 
        TODO: finish docstrings, input validation
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
    
    def _visualise_oriented_filters(self, thetas, sigma, save: bool=False, dest_dir: str=os.getcwd()): 
        """ Helper method; 
        TODO: finish docstrings, input validation, visualisation vmax/vmin
        """
        all_filters = []
        for angle in thetas:
            all_filters.append(self._get_oriented_filter(theta=angle,sigma=sigma))
        titles = [f'theta_{str(n)}' for n in thetas]
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
        else:
            plt.show();


    def meshwork_density(self):
        """ Accepts a binary image and returns the meshwork density.
        Uses the `skimage.measure()` function.  """
        if not (np.sort(np.unique(self.manipulated_stack)) == [0,1]).all():
            raise ValueError('Input image is not binary.')
        
        filled_image = morphology.binary_fill_holes(self.manipulated_stack)
        difference = filled_image - self.manipulated_stack
        n_holes = np.sum(difference)
        holes_percentage = n_holes*100 / (self.shape[0]*self.shape[1])
        self._call_hist('meshwork_density')
        raise NotImplementedError



    def meshwork_size(self):
        self._call_hist('meshwork_size')
        raise NotImplementedError



    def nuke(self):
        """ Restores an ActImg instance to it's pre-manipulated state.
        """
        self.manipulated_stack = None
        self.manipulated_depth = 0
        self.manipulated_substack_inds = None
        self._projected = None
        self._history = None
        return None


    def _call_hist(self, action):
        """ Helper records performed manipulations (called methods) in a list.
        """
        if self._history is None: 
            self._history = []
        self._history.append(action)
        return None



def get_ActinImg(image_name: str, image_dir: str):
    """ Creates an ActinImg instance. 
    """
    img, title = get_image_stack(os.path.join(image_dir, image_name), verbose=False)
    shape = img[0].shape
    depth = len(img)
    if depth == 1: 
        img = img[0]
    devonv = True if 'deconv' in image_name else False
    meta = get_meta(os.path.join(image_dir, image_name))
    resolut = get_resolution(meta)
    return ActinImg(np.asarray(img), title, shape, depth, devonv, resolut, meta)

    
