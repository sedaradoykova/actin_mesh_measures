import os
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import chain
from actin_meshwork_analysis.meshwork.actinimg import ActinImg, get_ActinImg
#from meshwork.utils import get_image_stack, list_all_tiffs, get_meta, get_resolution, list_files_dir_str

"""
- define what the focal plane is (done by Sabrina and Olivia, see excel.)
    - basal and cytoplasmic meshes
- more filling - find outline of cell and calculate 
    - maybe sequential with small steps? find outline
- size of mesh area is more important than percentage cell covered by mesh
"""

# im = get_image_stack(os.path.join(os.getcwd(), 'meshwork/test_data', '7min_Dual_fov2_deconv_1-3.tiff'), verbose=False)
# im[0][0].shape

# len(im)
# im[1]



""" Working out the mesh size. """

actimg = get_ActinImg('7min_Dual_fov2_deconv_1-3.tiff', os.path.join(os.getcwd(), 'meshwork/test_data'))
actimg.normalise()
actimg.z_project_min()
actimg.visualise('manipulated')
actimg.threshold(0.03)
actimg.visualise('manipulated')

from scipy.ndimage import correlate, morphology, label, binary_closing
from skimage import measure
img = np.copy(actimg.manipulated_stack)

closed_image = binary_closing(img, structure = np.ones((50,50)))
plt.imshow(closed_image, cmap='inferno'); plt.show();
difference2 = closed_image-img
plt.imshow(difference2, cmap='inferno'); plt.show();
labeled_image, num_features = label(closed_image)
print(np.bincount(labeled_image.ravel())[1:])

filled_image = morphology.binary_fill_holes(img)
plt.imshow(filled_image, cmap='inferno'); plt.show();
difference = filled_image-img
plt.imshow(difference, cmap='inferno'); plt.show();

np.sum(difference)
np.sum(difference)*100 / (difference.shape[0]*difference.shape[1])

labels = measure.label(img)
props = measure.regionprops(labels)

for prop in props:
    print('Label: {} >> Object size: {}'.format(prop.label, prop.area))

props[1].area

plt.imshow(labels, cmap=plt.cm.gnuplot); plt.show();

# actimg = get_ActinImg('7min_Dual_fov2_deconv_1-3.tiff', os.path.join(os.getcwd(), 'meshwork/test_data'))

# actimg.visualise('original', 1)
# actimg.normalise()
# actimg.z_project_min()
# actimg.threshold(0.015)
# actimg.visualise('manipulated')

# actimg.nuke()
# actimg.normalise()
# actimg.z_project_max()
# actimg.visualise('manipulated')


# meta = get_meta(os.path.join(os.getcwd(), 'test_data', 'CARs_11_11_22.lif - Fri_Beads.tif'))
# get_resolution(meta, True)
# meta.keys()

# meta['XResolution']
# meta['ImageDescription']

# 32903098/1000000




def z_proj_im():
    ones = np.ones(5)
    im = np.array([
        [*ones, *ones],[*ones, *ones],[*ones, *ones],
        [*ones*9, *ones*9],[*ones*5, *ones*5],[*ones*5, *ones*5],[*ones*9, *ones*9],
        [*ones, *ones],[*ones, *ones],[*ones, *ones]
        ])
    return np.array([im, np.transpose(im), np.arange(100).reshape(10,10)])


# pipeline works?
actimg = ActinImg(z_proj_im(), 'test.tiff', (10,10), 3, False, 20) 
#actimg.visualise_stack()
actimg.image_stack
actimg.normalise()
#actimg.visualise('manipulated')
actimg.manipulated_stack
actimg.z_project_max([0,1])
actimg.visualise('manipulated')

actimg.steerable_gauss_2order(None,2,360,True)
actimg.visualise_stack('manipulated', colmap='gray')

# bug in z-projection due to complex numbers (after steerable filter)
# what does np.real(x) do?? 
# try np.max(IM, axis=0)
actimg.z_project_min([0,1])
actimg.visualise('manipulated', colmap='gray')


# THERE MUST BE A BUG 
actimg = ActinImg(z_proj_im(), 'test.tiff', (10,10), 3, False, 20) 
actimg.normalise()
actimg.steerable_gauss_2order([0,2],2,360,True)
actimg = ActinImg(z_proj_im(), 'test.tiff', (10,10), 3, False, 20) 
actimg.normalise()
actimg.steerable_gauss_2order([1,2],2,360,True)
actimg = ActinImg(z_proj_im(), 'test.tiff', (10,10), 3, False, 20) 
actimg.normalise()
actimg.steerable_gauss_2order([2,2],2,360,True)



# import numpy as np 
# import matplotlib.pyplot as plt
# from meshwork.meshwork import ActinImg, get_ActinImg
# from meshwork.utils import get_image_stack, list_all_tiffs, get_meta, get_resolution, list_files_dir_str

# def z_proj_im():
#     ones = np.ones(5)
#     im = np.array([
# [*ones, *ones],[*ones, *ones],[*ones, *ones],[*ones*9, *ones*9],
# [*ones*5, *ones*5],[*ones*5, *ones*5],[*ones*9, *ones*9],[*ones, *ones],[*ones, *ones],[*ones, *ones]
# ])
#     return np.array([im, np.transpose(im)])


# actimg = ActinImg(z_proj_im(), 'test.tiff', (10,10), 2, False, 20) 

# import os
# actimg = get_ActinImg('7min_Dual_fov2_deconv_1-3.tiff', os.path.join(os.getcwd(), 'meshwork/test_data'))
# actimg.visualise_stack()
# actimg.normalise()
# actimg.z_project_max([1,2])
# actimg.visualise_stack('manipulated')

from scipy import io
avg_out = 'meshwork/steerable_gaussian/matlab_files/steerable_filters/steerable filters/average.mat'
mat = io.loadmat(avg_out)['Average']

plt.imshow(mat, cmap='gray'); plt.show();
np.max(mat)



impath = 'meshwork/steerable_gaussian/matlab_files/steerable_filters/steerable filters/'
actimg = get_ActinImg('testUTR_3minTHUfov1_04.tif', os.path.join(os.getcwd(), impath))
#actimg.visualise()
actimg.normalise()

actimg.steerable_gauss_2order(None,2,30,True)


actimg.visualise_stack('manipulated', colmap='gray')


np.histogram(actimg.image_stack)
plt.imshow(actimg.image_stack[0], cmap='gray', vmax=5e4);plt.show()


sigma, theta = 2, 30

theta = np.deg2rad(theta)

Wx = np.floor((8/2)*sigma)
Wx = Wx if Wx >= 1 else 1

x = np.arange(-Wx,Wx+1) ### tricky part because this chooses the output size??? 
xx,yy = np.meshgrid(x,x)


g0 = np.array(np.exp(-(xx**2+yy**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)))
G2a = np.array(-g0/sigma**2+g0*xx**2/sigma**4)
G2b =  np.array(g0*xx*yy/sigma**4)
G2c = np.array(-g0/sigma**2+g0*yy**2/sigma**4)
oriented_filter = (np.cos(theta))**2*G2a + np.sin(theta)**2*G2c - 2*np.cos(theta)*np.sin(theta)*G2b



for i, (im, title) in enumerate(zip([g0, G2a, G2b, G2c, oriented_filter], ['g0', 'G2a', 'G2b', 'G2c', 'oriented_filter'])):
    plt.subplot(1,5,i+1)
    plt.imshow(im, cmap='gray'); 
    plt.axis('off')
    plt.title(title)
plt.show();

gs = [g0, G2a, G2b, G2c]

min(list(map(np.min, gs)))
max(list(map(np.max, gs)))

# there are negative values in the gs which explains why we get negative values in the indecies 
# python doesn't support power operations for negative fractions
# hence the error
# the solution is to convert to complex numbers.... 






####################### GAUSS DEBUG

import os
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import chain
from actin_meshwork_analysis.meshwork.actinimg import ActinImg, get_ActinImg
#from meshwork.utils import get_image_stack, list_all_tiffs, get_meta, get_resolution, list_files_dir_str


impath = 'meshwork/'
actimg = get_ActinImg('testUTR_3miFRIifov1_01.tif', impath)
actimg.manipulated_stack = actimg.image_stack.copy()

#actimg.visualise()


actimg.steerable_gauss_2order(None,2,90,True)
actimg.visualise(imtype='manipulated')

actimg = get_ActinImg('testUTR_3miFRIifov1_01.tif', impath)
actimg.normalise()

#res_fig, res_stacks = [], [] 
res = [0]*6

for n, angle in enumerate(np.arange(0,360,60)):
    res[n] = actimg.steerable_gauss_2order(substacks=None, sigma=2, theta=angle, visualise=True)
    actimg.nuke()
    actimg.normalise()

mean = np.mean(np.asarray(res),0)
plt.imshow(mean, cmap='gray'); plt.show()


from scipy import io
j2_out = 'meshwork/steerable_gaussian/matlab_files/steerable_filters/steerable filters/average.mat'
mat = io.loadmat(j2_out)['J2']
j2_mean = np.mean(mat, 2)

plt.imshow(j2_mean, cmap='gray'); plt.show();


# shapes match 
j2_mean.shape == mean.shape

np.allclose(j2_mean, mean, atol=1e-20)

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
