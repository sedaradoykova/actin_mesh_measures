import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import correlate 
import cv2 

"""TODO fix correlation (commented out dysfunctional ones)
still don't understand why the noramalised images were working just fine. 
cv2.filter2D works with raw images (not normalised) but fails with normalised images
    input of raw images matches matlab 
    but normalised fails miserably with all 0s :( 
scipy.ndimage.correlate works with normalised images but not with raw data
    input of normalised matches matlab
    but raw shows weird behaviour, keeping image maximums 
"""


def steerable_gauss_2order(
    input_stack, substacks: list=None, sigma: float=2.0, theta: float=0., visualise: bool=True,
    return_stack: bool=False, return_dict: bool=False):
    """ Steers an X-Y separable second order Gaussian filter in direction theta.
    Implemented according to W. T. Freeman and E. H. Adelson, "The Design and Use of Steerable Filters", IEEE PAMI, 1991.
    Based on matlab code from Jincheng Pang, Tufts University, 2013.
    ...
    Arguments 
    ---------
    substacks : list
        A list of length=2, specifying the range [start_index, finish_index] of slices to perform the operation on.
        Note: Python indexing starts at 0. 
    sigma : float
        The standard deviation of the Gaussian.
    theta : float
        The steerable filter orientation. 

    Returns
    -------
    response : numpy array 
        The response of the theta-rotated derivative. 
    """
    if not isinstance(input_stack, (tuple, np.ndarray)): 
        raise ValueError('Input ')
    if substacks: 
        data = input_stack[substacks[0]:substacks[1]+1].copy()
    else:
        data = input_stack.copy()

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
    G2b = np.array(g0*xx*yy/sigma**4)
    G2c = np.array(-g0/sigma**2+g0*yy**2/sigma**4)

    oriented_filter = np.array((np.cos(theta))**2*G2a + np.sin(theta)**2*G2c - 2*np.cos(theta)*np.sin(theta)*G2b)

    if len(data.shape) == 3: 
        depth, rows, cols = data.shape
    else:
        depth, rows, cols = 1, *data.shape
    response_stack = np.empty((depth, rows, cols))
    response_stack.fill(np.nan)
    if depth > 1:
        for count, input in enumerate(data): 
            #### Oriented filter response.
            # Calculate image gradients, using separability.
            # I2a = correlate(input, G2a, mode='nearest')
            # I2b = correlate(input, G2b, mode='nearest')
            # I2c = correlate(input, G2c, mode='nearest')
            I2a = cv2.filter2D(input, -1, G2a, borderType=cv2.BORDER_REPLICATE)
            I2b = cv2.filter2D(input, -1, G2b, borderType=cv2.BORDER_REPLICATE)
            I2c = cv2.filter2D(input, -1, G2c, borderType=cv2.BORDER_REPLICATE)

            # Evaluate oriented filter response. 
            response = np.array(
                (np.cos(theta))**2*I2a + np.sin(theta)**2*I2c - 2*np.cos(theta)*np.sin(theta)*I2b, dtype='uint16')
            response_stack[count,:,:] = np.copy(response)
    else:
        input = data
        # I2a = correlate(input, G2a, mode='nearest')
        # I2b = correlate(input, G2b, mode='nearest')
        # I2c = correlate(input, G2c, mode='nearest')
        I2a = cv2.filter2D(input, -1, G2a, borderType=cv2.BORDER_REPLICATE)
        I2b = cv2.filter2D(input, -1, G2b, borderType=cv2.BORDER_REPLICATE)
        I2c = cv2.filter2D(input, -1, G2c, borderType=cv2.BORDER_REPLICATE)

        # Evaluate oriented filter response.
        response_stack[0,:,:] = np.array(
            (np.cos(theta))**2*I2a + np.sin(theta)**2*I2c - 2*np.cos(theta)*np.sin(theta)*I2b, dtype='uint16')

    # Visualise 
    if visualise: 
        figrows, figcols, left_to_del = 1, depth+2, None
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
                axes.flatten(), [data, response_stack[0], oriented_filter], 
                [*['Input']*depth, *['Response']*depth, 'Oriented filter']
                ):
                ax.imshow(image, cmap='gray')
                ax.set_axis_off()
                ax.set_title(title)
        if left_to_del:
            for ind1, ind2 in left_to_del:
                fig.delaxes(axes[ind1][ind2])

        plt.show();

    if return_dict: 
        keys_out = ['g0', 'G2a', 'G2b', 'G2c', 'oriented_filter', 'I2a', 'I2b', 'I2c']
        vals = [g0, G2a, G2b, G2c, oriented_filter, I2a, I2b, I2c]
        res_dict = {key: val for key, val in zip(keys_out, vals)}
        res_dict['J'] = response_stack[0]
        res_dict['I'] = data
        return res_dict

    if return_stack:
        return response_stack[0]
