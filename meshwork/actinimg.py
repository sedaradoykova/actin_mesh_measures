import os
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
        - [ ] test steerable second order filter but how?
        - [ ! ] BUG in steerable filter: inconsistent results (see scratch) 
        - [ ! ] BUG in z_proj_min/max: induced by complex numbers
        - when to return self?   
        - think about selecting particular frames instead of specifying ranges (projection, plotting)...
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
            raise TypeError('type must be a string and ind must be an integer')
        if (scale_bar and self.resolution is None):
            raise ValueError('Resolution has not been specified.')
        if imtype=='manipulated':
            if self.manipulated_stack is None: 
                raise ValueError('Raw data has not been normalised.')
            if ind < 1 or ind > self.manipulated_depth:
                raise ValueError(f'ind must be an integer in range (1, {self.manipulated_depth}')        
            if self._projected or self.manipulated_depth == 1:
                plt.imshow(self.manipulated_stack, cmap=colmap)
            else: 
                plt.imshow(self.manipulated_stack[ind-1], cmap=colmap)        
        elif imtype=='original':
            if ind < 1 or ind > self.depth:
                raise ValueError(f'ind must be an integer in range (1, {self.depth}')  
            if self.depth > 1:
                plt.imshow(self.image_stack[ind-1], cmap=colmap)
            else: 
                plt.imshow(self.image_stack, cmap=colmap)
        else: 
            raise ValueError('Image type \'{imtype}\' not recognised; type must be one of  [\'original\', \'manipulated\']'.format(imtype=imtype))
        if (scale_bar and self.resolution is not None):
            scalebar = ScaleBar(self.resolution, 'nm', box_color='None', color='#F2F2F2', location=bar_locate) 
            plt.gca().add_artist(scalebar)
        plt.axis('off')
        if save: 
            imtitle = f'{self.title}_{"_".join(self._history)}' if self._history else self.title
            dest = os.path.join(dest_dir, imtitle, '.png')
            plt.imsave(dest)
        else:
            plt.show();


    def visualise_stack(
        self, imtype: str='original', substacks: list=None, save: bool=False, dest_dir : str=os.getcwd(), 
        fig_size: tuple=None, colmap: str='inferno', scale_bar: bool=True, bar_locate:str='upper left'
        ):
        """ Visualise a stack as a tile of all constituent images. 
        ...
        Arguments 
        ---------
        imtype : str='original'
            A string specifying whether to display `original` or `modified` data. 
        substacks : list
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
            raise TypeError('type must be a string.')
        if substacks and (len(substacks) != 2 or not isinstance(substacks, list)): 
            raise ValueError('Substacks has to be a list of length=2, specifying a range.')
        if imtype=='original':
            if substacks[0] < 1 or substacks[1] > self.depth:
                raise ValueError(f'Substacks must be integers in range (1, {self.manipulated_depth}).')
            if substacks: 
                data = self.image_stack[substacks[0]-1:substacks[1]].copy() 
                figrows, figcols = get_fig_dims(substacks[1]-substacks[0]+1)
            else:
                data = self.image_stack.copy()
                figrows, figcols = get_fig_dims(self.depth) 
                
            absmin, absmax = np.min(data), np.max(data)
            for n, image in enumerate(data[::1]):
                ax = plt.subplot(figrows,figcols,n+1)
                ax.imshow(image, cmap=colmap, vmin=absmin, vmax=absmax)
                if (scale_bar and self.resolution is not None):
                    scalebar = ScaleBar(self.resolution, 'nm', box_color='None', color='#F2F2F2', location=bar_locate) 
                    ax.add_artist(scalebar)
                ax.set_axis_off()
                ax.text(0.66, 0.05, 'n={count}'.format(count=substacks[0]+n),color='#F2F2F2',transform=ax.transAxes)

        elif imtype=='manipulated':
            if self.manipulated_stack is None: 
                raise ValueError('Raw data has not been normalised.')
            if self._projected:
                self.visualise('manipulated')
            else:
                if substacks[0] < 1 or substacks[1] > self.depth:
                    raise ValueError(f'Substacks must be integers in range (1, {self.manipulated_depth}).')
                if substacks: 
                    data = self.manipulated_stack[substacks[0]-1:substacks[1]].copy() 
                    figrows, figcols = get_fig_dims(substacks[1]-substacks[0]+1)
                else:
                    data = self.manipulated_stack.copy()
                    manipulated_depth = data.shape[0]
                    figrows, figcols = get_fig_dims(manipulated_depth) 

                absmin, absmax = np.min(data), np.max(data)
                for n, image in enumerate(data[::1]):
                    ax = plt.subplot(figrows,figcols,n+1)
                    ax.imshow(image, cmap=colmap, vmin=absmin, vmax=absmax)
                    if (scale_bar and self.resolution is not None):
                        scalebar = ScaleBar(self.resolution, 'nm', box_color='None', color='#F2F2F2', location=bar_locate) 
                        ax.add_artist(scalebar)
                    ax.set_axis_off()
                    ax.text(0.66, 0.05, 'n={count}'.format(count=substacks[0]+n),color='#F2F2F2',transform=ax.transAxes)

        else:
            raise ValueError('Image type \'{imtype}\' not recognised; type must be one of  [\'original\', \'manipulated\']'.format(imtype=imtype))

        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        if save==True: 
            imtitle = f'{self.title}_{"_".join(self._history)}' if self._history else self.title
            dest = os.path.join(dest_dir, imtitle, '.png')
            plt.imsave(dest)
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

 
    def z_project_min(self, substacks: list[int]=None):
        """ Returns the minimum axial projection of a z-stack. Retains the minimum intensity at every position.
        Can be performed on a sub-range.
        ...
        Arguments 
        ----------
        substacks : list
            A list of length=2, specifying the range [start_index, finish_index] of slices to perform the operation on.
            Note: indexing is from 1 to `n_frames` in image stack. 
        See also
        --------
        ActImg.z_project_max()
        """
        if self.manipulated_stack is None: 
            raise ValueError('Raw data has not been normalised.')
        if substacks and (len(substacks) != 2 or not isinstance(substacks, list)): 
            raise ValueError('Substacks has to be a list of length=2, specifying a range.')
        if substacks[0] < 1 or substacks[1] > self.manipulated_depth:
            raise ValueError(f'Substacks must be integers in range (1, {self.manipulated_depth}).')
        if substacks: 
            data = self.manipulated_stack[substacks[0]-1:substacks[1]].copy() 
        else:
            data = self.manipulated_stack.copy()
        depth, width, height = data.shape
        
        flat_res = [min(row) for row in np.transpose(np.array(data).ravel().reshape((depth,width*height)))]
        self.manipulated_stack = np.array(flat_res).reshape(*self.shape)
        self.manipulated_depth = 1
        self._projected = True
        self._call_hist('z_project_min')
        return None


    def z_project_max(self, substacks: list[int]=None):
        """ Returns the maximum axial projection of a z-stack. Retains the maximum intensity at every pixel position. 
        Can be performed on a sub-range.
        ...
        Arguments 
        ----------
        substacks : list
            A list of length=2, specifying the range [start_index, finish_index] of slices to perform the operation on.
            Note: indexing is from 1 to `n_frames` in image stack. 
        See also
        --------
        ActImg.z_project_min()   
        """
        if self.manipulated_stack is None: 
            raise ValueError('Raw data has not been normalised.')
        if substacks and (len(substacks) != 2 or not isinstance(substacks, list)): 
            raise ValueError('Substacks has to be a list of length=2, specifying a range.')
        if substacks[0] < 1 or substacks[1] > self.manipulated_depth:
            raise ValueError(f'Substacks must be integers in range (1, {self.manipulated_depth}).')
        if substacks: 
            data = self.manipulated_stack[substacks[0]-1:substacks[1]].copy()
        else:
             data = self.manipulated_stack.copy()
        depth, width, height = data.shape

        flat_res = [max(row) for row in np.transpose(np.array(data).ravel().reshape((depth,width*height)))]
        self.manipulated_stack = np.array(flat_res).reshape(*self.shape)
        self.manipulated_depth = 1
        self._projected = True
        self._call_hist('z_project_max')
        return None


    def threshold(self, threshold: float):
        """ Returns a binary thresholded image. 
        """
        if self.manipulated_stack is None: 
            raise ValueError('Raw data has not been normalised.')
        if self._projected is None: 
            raise ValueError('Data has not been projected in z.')
        if threshold < 0: 
            raise ValueError('Threshold cannot be negative.')
        self.manipulated_stack = np.array([1 if p > threshold else 0 for p in self.manipulated_stack.ravel()]).reshape(*self.shape)
        self._call_hist('threshold')
        return None

        
    def steerable_gauss_2order(self, substacks: list=None, sigma: float=2.0, theta: float=0., visualise: bool=True):
        """ Steers an X-Y separable second order Gaussian filter in direction theta.
        Implemented according to W. T. Freeman and E. H. Adelson, "The Design and Use of Steerable Filters", IEEE PAMI, 1991.
        Based on matlab code from Jincheng Pang, Tufts University, 2013.
        ...
        Arguments 
        ---------
        substacks : list
            A list of length=2, specifying the range [start_index, finish_index] of slices to perform the operation on.
            Note: indexing is from 1 to `n_frames` in image stack. 
        sigma : float
            The standard deviation of the Gaussian.
        theta : float
            The steerable filter orientation. 

        Returns
        -------
        response : numpy array 
            The response of the theta-rotated derivative. 
        """
        if self.manipulated_stack is None: 
            raise ValueError('Raw data has not been normalised.')
        if substacks and (len(substacks) != 2 or not isinstance(substacks, list)): 
            raise ValueError('Substacks has to be a list of length=2, specifying a range.')
        if substacks[0] < 1 or substacks[1] > self.manipulated_depth:
            raise ValueError(f'Substacks must be integers in range (1, {self.manipulated_depth}).')
        if substacks: 
            data = self.manipulated_stack[substacks[0]-1:substacks[1]].copy()
        else:
            data = self.manipulated_stack.copy()

        #### Separable filter kernels
        # Gaussian filter mesh grid 
        Wx = np.floor((8/2)*sigma)
        Wx = Wx if Wx >= 1 else 1

        x = np.arange(-Wx,Wx+1) # determines kernel size 
        xx,yy = np.meshgrid(x,x)

        theta = np.deg2rad(-theta) # convert to radians, clockwise

        # Filter kernels 
        g0 = np.array(np.exp(-(xx**2+yy**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)))
        G2a = np.array(-g0/sigma**2+g0*xx**2/sigma**4)
        G2b =  np.array(g0*xx*yy/sigma**4)
        G2c = np.array(-g0/sigma**2+g0*yy**2/sigma**4)

        oriented_filter = np.array((np.cos(theta))**2*G2a + np.sin(theta)**2*G2c - 2*np.cos(theta)*np.sin(theta)*G2b)
        
        if self.manipulated_depth > 1: 
            depth, rows, cols = data.shape
        else:
            depth, rows, cols = 1, *self.shape
        response_stack = np.empty((depth, rows, cols))
        response_stack.fill(np.nan)
        if depth > 1:
            for count, input in enumerate(data): 
                #### Oriented filter response.
                # Calculate image gradients, using separability.
                I2a = correlate(input, G2a, mode='nearest')
                I2b = correlate(input, G2b, mode='nearest')
                I2c = correlate(input, G2c, mode='nearest')

                # Evaluate oriented filter response.
                response = np.array(
                    (np.cos(theta))**2*I2a + np.sin(theta)**2*I2c - 2*np.cos(theta)*np.sin(theta)*I2b)
                response_stack[count,:,:] = np.copy(response)
        else:
            input = data.copy()
            I2a = correlate(input, G2a, mode='nearest')
            I2b = correlate(input, G2b, mode='nearest')
            I2c = correlate(input, G2c, mode='nearest')

            # Evaluate oriented filter response.
            response = np.array(
                (np.cos(theta))**2*I2a + np.sin(theta)**2*I2c - 2*np.cos(theta)*np.sin(theta)*I2b)
            response_stack = np.copy(response)

        if visualise: 
            figrows, figcols, left_to_del = get_fig_dims(depth*2+1) 
            ratio = figrows/figcols
            fig, axes = plt.subplots(nrows=figrows, ncols=figcols, figsize=(10*ratio,10*ratio))

            if depth > 1:
                for ax, image, title in zip(
                    axes.flatten(), 
                    [*np.rollaxis(data, 0), *np.rollaxis(response_stack, 0), oriented_filter], 
                    [*['Input']*depth, *['Response']*depth, 'Oriented filter']
                    ):
                    ax.imshow(image, cmap='gray')
                    ax.set_axis_off()
                    ax.set_title(title)
            else: 
                for ax, image, title in zip(
                    axes.flatten(), [data, response_stack, oriented_filter], 
                    [*['Input']*depth, *['Response']*depth, 'Oriented filter']
                    ):
                    ax.imshow(image, cmap='gray')
                    ax.set_axis_off()
                    ax.set_title(title)
            if left_to_del:
                for ind1, ind2 in left_to_del:
                    fig.delaxes(axes[ind1][ind2])

            plt.show();

        self.manipulated_stack = response_stack
        self.manipulated_depth = depth
        self._call_hist('steerable_gauss_2order')
        return response_stack

    
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

    
