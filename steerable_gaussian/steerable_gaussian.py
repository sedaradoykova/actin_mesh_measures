from scipy.ndimage import correlate 
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image


# test_im = np.asarray(Image.open('meshwork/steerable_gaussian/test_image.png').convert('L'))
# test_im.shape
# plt.imshow(test_im, cmap='gray'); plt.show();


test_im = np.asarray(Image.open('meshwork/steerable_gaussian/matlab_files/steerable_filters/steerable filters/example.png'))#.convert('L'))

def steerGaussFilterOrder2(img: np.ndarray, theta: float=0., sigma: float=2.0, visualise: bool=True):
    """ X-Y separable second order Gaussian steerable filter. 
    According to W. T. Freeman and E. H. Adelson, "The Design and Use of Steerable Filters", IEEE PAMI, 1991.
    Based on matlab code from Jincheng Pang, Tufts University, 2013.
    ...
    Arguments 
    ---------
    img : numpy array
        Input image.
    theta : float
        The steerable filter orientation. 
    sigma : float
        The standard deviation of the Gaussian.
    Returns
    -------
    response : numpy array 
        The response of the theta-rotated derivative. 
    """
    # original code takes the mean along third axis (RGB image) to yield a grayscale image
    # if len(img.shape) == 3:
    #     img = np.mean(img,2)

    theta = np.deg2rad(-theta)

    #### Separable filter kernels.
    # Gaussian filter meshgrid 
    Wx = np.floor((8/2)*sigma)
    Wx = Wx if Wx >= 1 else 1

    x = np.arange(-Wx,Wx+1) # determines kernel size sqrt
    xx,yy = np.meshgrid(x,x)

    # Filter kernels 
    g0 = np.array(np.exp(-(xx**2+yy**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)))
    G2a = np.array(-g0/sigma**2+g0*xx**2/sigma**4)
    G2b = np.array(g0*xx*yy/sigma**4)
    G2c = np.array(-g0/sigma**2+g0*yy**2/sigma**4)

    #### Oriented filter response.
    # Calculate image gradients, using separability.
    I2a = correlate(img, G2a, mode='nearest')
    I2b = correlate(img, G2b, mode='nearest')
    I2c = correlate(img, G2c, mode='nearest')

    # Evaluate oriented filter response.
    response = np.array(
        (np.cos(theta))**2*I2a + np.sin(theta)**2*I2c - 2*np.cos(theta)*np.sin(theta)*I2b)
    oriented_filter = np.array((np.cos(theta))**2*G2a + np.sin(theta)**2*G2c - 2*np.cos(theta)*np.sin(theta)*G2b)

    # Visualise 
    if visualise: 
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9,3))
        for ax, image, title in zip(
            axes.flatten(), [img, response, oriented_filter], ['Input', 'Response', 'Oriented filter']):
            ax.imshow(image, cmap='gray')
            ax.set_axis_off()
            ax.set_title(title)
        plt.show();
    #return response

# steerGaussFilterOrder2(test_im, 360, 2)
# steerGaussFilterOrder2(test_im, 90, 2)
# steerGaussFilterOrder2(test_im, 30, 2)
# steerGaussFilterOrder2(test_im, 270, 2)

from tifffile import imread
# test_im = np.asarray(Image.open('example.png'))#.convert('L'))    # png image 
impath = 'meshwork/testUTR_3miFRIifov1_01.tif'                      # tiff file from deconvolved data 
#impath = 'meshwork/testUTR_3minTHUfov1_04.tif'
test_im = imread(impath)
#plt.imshow(test_im, cmap='gray'); plt.show();

steerGaussFilterOrder2(test_im, sigma=2, theta=90)


normalised = (test_im-np.min(test_im))/(np.max(test_im)-np.min(test_im))

#res_fig, res_stacks = [], [] 
res = [0]*6

for n, angle in enumerate(np.arange(0,360,60)):
    res[n] = steerGaussFilterOrder2(normalised, sigma=2, theta=angle)

mean = np.mean(np.asarray(res),0)
plt.imshow(mean, cmap='gray'); plt.show()


from scipy import io
j2_out = 'meshwork/steerable_gaussian/matlab_files/steerable_filters/steerable filters/average.mat'
mat = io.loadmat(j2_out)['J2']
j2_mean = np.mean(mat, 2)

plt.imshow(j2_mean, cmap='gray'); plt.show();


# shapes match 
j2_mean.shape == mean.shape

np.allclose(j2_mean, mean, atol=1e-8)

minmin = np.min([np.min(j2_mean), np.min(mean)])
maxmax = np.max([np.max(j2_mean), np.max(mean)])

plt.figure(figsize=(10,4))
plt.subplots_adjust(hspace=0.01, vspace=0, top=0)
plt.subplot(1,3,1)
plt.imshow(j2_mean, cmap='gray', vmax=maxmax, vmin=minmin)
plt.title('matlab output')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(mean, cmap='gray', vmax=maxmax, vmin=minmin)
plt.title('python output')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(mean-j2_mean,cmap='gray', vmax=maxmax, vmin=minmin)
plt.title('difference')
plt.axis('off')
plt.suptitle('comparing output for averaging over $\\theta$=[0,60,120,180,240,300]\nnp.allclose(matlab_mean, pyhton_mean, atol=1e-8)==True', fontsize=15)
plt.show();
