"""
x or x or X (both)
File name                              Type    Basal     Cytosolic  new surf, um^2   Fixed?
1min_Thurs_FOV2_decon.tif       Untransduced     [1] X    [4, 6] X   24.45; 35.86     V V
1min_Thurs_FOV4_decon_left.tif  Untransduced  [1, 3] x    [6, 8] -    9.83            V
1min_Thurs_FOV4_decon_right.tif Untransduced     [1] -    [3, 6] x   ----             X   connected to other cell in corner - could work without that?
1min_Thurs_FOV5_decon.tif       Untransduced  [1, 2] x    [4, 6] -   34.78            V
1min_Thurs_FOV6_decon.tif       Untransduced     [1] x    [4, 6] -   14.62            V   but am i introducing artifacts? not v clear what 1 min cell boundaries are 
3min_Thurs_FOV4_decon.tif       Untransduced     [1] X    [4, 6] X   16.92; 31.99     V V
8min_Thurs_FOV3_decon.tif       Untransduced  [2, 4] x    [6, 8] -   192.46           V   beautiful example of getting entire cell area
1min_FOV3_decon_left.tif        CAR_dual         [1] x    [4, 6] -   126.13           V
1min_FOV4_decon_left.tif        CAR_dual      [1, 2] x    [5, 7] -   ----             X   broken/cropped boundary
1min_FOV5_decon.tif             CAR_dual      [2, 3] x    [6, 8] -   210.71           V   good example of V 
1min_FOV8_decon_bottom_left.tif CAR_dual         [1] X    [3, 4] X   --------         X X     struggle with cropped cells
1min_FOV8_decon_right.tif       CAR_antiCD22     [1] X    [3, 4] X   --------         X X     struggle with cropped cells
1min_FOV9_decon_top_right.tif   CAR_dual         [1] -    [3, 5] x   67.14            V   works well with inner ring (clearance)
3min_FOV1_decon_left.tif        CAR_dual      [2, 4] x    [7, 9] -   ----             X   inner ring bc outer is interrupted
3min_FOV4_decon_top_right.tif   CAR_antiCD19     [1] -    [3, 5] x   ----             X   would it work if for cytosolic (expect clearance) = expect two rings? 
8min_FOV1_decon_bottom_right.tif CAR_antiCD19 [7, 9] -  [10, 11] x   ----             X   connected to other cell in corner - could work without that?
8min_FOV1_decon_top_left.tif    CAR_antiCD22  [7, 9] x  [10, 11] -   ----             X   connected to x2 other cell in corner - could work without that?
8min_FOV6_decon_top.tif         CAR_antiCD19  [1, 3] X    [5, 7] X   --------         X X     struggle with cropped cells
                                                                                        Solved = 11; Not solved = 12 
Surface area (um^2)  =  0.36
Image not segmented well: 1min_FOV8_decon_bottom_left.tif
Surface area (um^2)  =  0.04
Image not segmented well: 1min_FOV8_decon_bottom_left.tif
Surface area (um^2)  =  1.44
Image not segmented well: 1min_FOV8_decon_right.tif
Surface area (um^2)  =  0.98
Image not segmented well: 1min_FOV8_decon_right.tif
Surface area (um^2)  =  5.48
Image not segmented well: 8min_FOV6_decon_top.tif
Surface area (um^2)  =  3.46
Image not segmented well: 8min_FOV6_decon_top.tif
Surface area (um^2)  =  0.16
Image not segmented well: 1min_Thurs_FOV4_decon_right.tif
Surface area (um^2)  =  1.50
Image not segmented well: 1min_FOV4_decon_left.tif
Surface area (um^2)  =  20.98
Image not segmented well: 3min_FOV1_decon_left.tif
Surface area (um^2)  =  0.03
Image not segmented well: 3min_FOV4_decon_top_right.tif
Surface area (um^2)  =  0.21
Image not segmented well: 8min_FOV1_decon_bottom_right.tif
Surface area (um^2)  =  0.32
Image not segmented well: 8min_FOV1_decon_top_left.tif
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from meshure.actimg import get_ActImg
from scipy.ndimage import correlate, morphology, label, binary_closing
from scipy import ndimage 
from skimage import measure
from skimage.morphology import convex_hull_image, convex_hull_object 
import skimage.morphology as morpho # skimorph


def surface_area(actimg, vis=False):
    """ 
    Returns the surface area and associated units. 
    Note: the algorithm uses serial dilations to include the periphery of the cell even if the cell boundary is discontinuous.
    The dilations are followed by an equivalent number of serial erosions to avoid overestimating the cell area. 
    Note: it is assumed that the largest object in the field of view is the cell which is to be segmented.
    Returns
    -------
    if the cell is not segmented, this is recorded separately

    """
    img = np.copy(actimg.manipulated_stack)
    # close
    closed_image = binary_closing(img, structure = np.ones((5,5))) # compare    5,5   7,7   10,10   15,15   30,30
    # dilate serially
    dilated = morpho.dilation(closed_image).astype('int')
    dilated = morpho.dilation(dilated).astype('int')
    dilated = morpho.dilation(dilated).astype('int')
    # fill
    filled_image = morphology.binary_fill_holes(dilated)
    # find contour
    contours = measure.find_contours(filled_image)
    ind_max = np.argmax([len(x) for x in contours])
    cont_img = np.zeros(filled_image.shape)
    cont_img[np.ceil(contours[ind_max]).astype('int')[:,0], np.ceil(contours[ind_max]).astype('int')[:,1]] = 1
    # dilate contour 
    dilated_cont = morpho.dilation(cont_img).astype('int')
    # fill contour 
    filled_cont = morphology.binary_fill_holes(dilated_cont).astype('int')
    # erode serially (equal to no. dilations)
    eroded_cont = morpho.binary_erosion(filled_cont).astype('int')
    eroded_cont = morpho.binary_erosion(eroded_cont).astype('int')
    eroded_cont = morpho.binary_erosion(eroded_cont).astype('int')

    if vis:
        plt.imshow(img)
        plt.imshow(eroded_cont, cmap='gray', alpha=0.5)
        plt.title(f'{imname} : {sub}')
        plt.show()

    surface_area = np.sum(eroded_cont)*(actimg.resolution['pixel_size_xy']**2)/1e6
    print(f'Surface area (um^2)  =  {surface_area:.2f}')
    if surface_area < np.sum(img)*(actimg.resolution['pixel_size_xy']**2)/1e6 - surface_area:
        print(f'Image not segmented well: {actimg.title}')


data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/deconv_data/")

all_paths = [['Untransduced_1.11.22_processed_imageJ', '1min_Thurs_FOV2_decon.tif'],
             ['Untransduced_1.11.22_processed_imageJ', '3min_Thurs_FOV4_decon.tif'],
             ['CARs_8.11.22_processed_imageJ', '1min_FOV8_decon_bottom_left.tif'],
             ['CARs_8.11.22_processed_imageJ', '1min_FOV8_decon_right.tif'],
             ['CARs_8.11.22_processed_imageJ', '8min_FOV6_decon_top.tif'],
             ['Untransduced_1.11.22_processed_imageJ', '1min_Thurs_FOV4_decon_left.tif'],
             ['Untransduced_1.11.22_processed_imageJ', '1min_Thurs_FOV4_decon_right.tif'],
             ['Untransduced_1.11.22_processed_imageJ', '1min_Thurs_FOV5_decon.tif'],
             ['Untransduced_1.11.22_processed_imageJ', '1min_Thurs_FOV6_decon.tif'],
             ['Untransduced_1.11.22_processed_imageJ', '8min_Thurs_FOV3_decon.tif'],
            ['CARs_8.11.22_processed_imageJ', '1min_FOV3_decon_left.tif'],
            ['CARs_8.11.22_processed_imageJ', '1min_FOV4_decon_left.tif'],
            ['CARs_8.11.22_processed_imageJ', '1min_FOV5_decon.tif'],
            ['CARs_8.11.22_processed_imageJ', '1min_FOV9_decon_top_right.tif'],
            ['CARs_8.11.22_processed_imageJ', '3min_FOV1_decon_left.tif'],
            ['CARs_8.11.22_processed_imageJ', '3min_FOV4_decon_top_right.tif'],
            ['CARs_8.11.22_processed_imageJ', '8min_FOV1_decon_bottom_right.tif'],
            ['CARs_8.11.22_processed_imageJ', '8min_FOV1_decon_top_left.tif'],
             ]
all_planes = [ 
    [[1],[4, 6]], [[1],[4, 6]], [[1],[3, 4]], [[1],[3, 4]], [[1, 3],[5, 7]],
    [[1, 3], None], [None, [3, 6]], [[1, 2], None], [[1], None], [[2, 4], None],
    [[1], None], [[1, 2], None], [[2, 3], None], [None, [3, 5]], 
    [[2, 4], None], [None, [3, 5]], [None, [10, 11]], [[7, 9], None]]


all_paths, all_planes = all_paths[0:2], all_planes[0:2]

for ((path_in, imname), (base, cyto)) in zip(all_paths, all_planes):
    for sub in (base, cyto):
        if sub: 
            actimg = get_ActImg(imname, os.path.join(data_path, path_in)) 
            actimg.normalise()
            actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0,180,20),sigma=2,substack=sub,visualise=False)
            actimg.z_project_min()
            actimg.threshold_dynamic(std_dev_factor=0, return_mean_std_dev=False)

            actimg.meshwork_density(False)
            actimg.meshwork_size(False, False, False) #FFT
            actimg.surface_area(False)

            surface_area(actimg, vis=True)
            # actimg.save_estimated_params()
            # actimg._history

            # actimg.estimated_parameters.keys()
            # actimg.estimated_parameters['mesh_size_summary']
            # actimg.estimated_parameters['aggregated_line_profiles']
            # actimg.estimated_parameters['surface_area'] / 1e6

actimg = get_ActImg('1min_FOV3_decon_left.tif', os.path.join(data_path, 'CARs_8.11.22_processed_imageJ')) 
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0,180,20),sigma=2,substack=[1],visualise=False)
actimg.z_project_min()
actimg.threshold_dynamic(std_dev_factor=0, return_mean_std_dev=False)

actimg.meshwork_density(False)
actimg.meshwork_size(False, False, False) #FFT
actimg.surface_area(False)

imname, sub = '', ''
surface_area(actimg, vis=True)

import pandas as pd

newim = np.zeros(actimg.shape)
inds = np.nonzero(actimg._binary_images['cell_surface'])
contours = measure.find_contours(actimg._binary_images['cell_surface'])
contour = contours[np.argmax([len(x) for x in contours])]
contour_ceil = np.ceil(contour).astype('int')
contour_floor = np.floor(contour).astype('int')

newim[inds] = actimg.manipulated_stack[inds].copy()
newim[contour_ceil[:,0], contour_ceil[:,1]] = 1
newim[contour_floor[:,0], contour_floor[:,1]] = 1
newim = binary_closing(newim, structure = np.ones((1,2)))
newim = binary_closing(newim, structure = np.ones((2,1)))
newimg_inverted = (newim==0).astype('int')

labels = measure.label(newimg_inverted, connectivity=1) # 4-connectivity for 2d images
equiv_diams = pd.DataFrame(measure.regionprops_table(labels, actimg._binary_images['cell_surface'], properties=['equivalent_diameter_area']))*actimg.resolution['pixel_size_xy']

plt.subplot(1,2,1)
plt.imshow(actimg._binary_images['cell_surface'])
plt.imshow(actimg.manipulated_stack, cmap='gray', alpha=0.1)
plt.axis('off'); plt.title('segmented cell surface')
plt.subplot(1,2,2)
plt.imshow(labels, cmap='tab20c_r')
plt.imshow(newim, cmap='gray', alpha=0.1)
plt.title('inverted image labels')
plt.axis('off'); plt.show()


plt.imshow(newimg_inverted)
plt.show()


actimg.meshwork_size(False, False, True)
"""
Artifact correction: 
skimage.segmentation.clear_border(labels, buffer_size=0, bgval=0, mask=None, *, out=None)
    use to remove labels which touch border? 
    pros: will remove some artifacts and cropped excess cells
    cons: cells touching border often overlap and sometimes mesh is not captured entirely in the field of view  
"""


# Explore other segmentation methods 

img = np.copy(actimg.manipulated_stack)
img_inverted = (img==0).astype('int')

# closing 
closed_image = binary_closing(img, structure = np.ones((5,5))) # compare    5,5   7,7   10,10   15,15   30,30
difference_closed = ~closed_image+img
# filling 
filled_image = morphology.binary_fill_holes(img)#,structure=np.ones((5,5)))
filled_image = morphology.binary_fill_holes(closed_image)
difference_filled = ~filled_image+img


plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title('binary image')
plt.axis('off')
plt.subplot(2,3,2)
plt.imshow(closed_image, cmap='gray')
plt.title('closed image')
plt.axis('off')
plt.subplot(2,3,3)
plt.imshow(difference_closed, cmap='gray')
plt.title('difference image')
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(img, cmap='gray')
plt.title('binary image')
plt.axis('off')
plt.subplot(2,3,5)
plt.imshow(filled_image, cmap='gray')
plt.title('filled image')
plt.axis('off')
plt.subplot(2,3,6)
plt.imshow(difference_filled, cmap='gray')
plt.title('difference image')
plt.axis('off')
plt.show();

# labels = measure.label(img_inverted)#, connectivity=0.5)
# plt.imshow(closed_image, cmap='tab20c_r'); plt.show();


# find contour

contours = measure.find_contours(filled_image)
fig, ax = plt.subplots()
ax.imshow(filled_image)
ind_max = np.argmax([len(x) for x in contours])
ax.plot(contours[ind_max][:, 1], contours[ind_max][:, 0], linewidth=2)
ax.axis('Image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

from skimage.morphology import convex_hull_image, convex_hull_object 
import skimage.morphology as morpho 

cont_img = np.zeros(filled_image.shape)
cont_img[np.ceil(contours[ind_max]).astype('int')[:,0], np.ceil(contours[ind_max]).astype('int')[:,1]] = 1
# for contour in contours: 
#     cont_img[np.ceil(contour).astype('int')[:,0], np.ceil(contour).astype('int')[:,1]] = 1

plt.imshow(cont_img); plt.show()



chull = convex_hull_object(cont_img)
plt.imshow(chull)
plt.imshow(img, alpha=0.5)
plt.show()

centroid = np.array(list(measure.regionprops_table(chull.astype('int'), properties=['centroid']).values())).astype('int').reshape(2,)

dilated = morpho.dilation(cont_img).astype('int')
filled = morphology.binary_fill_holes(dilated).astype('int')
eroded = morpho.binary_erosion(filled).astype('int')
plt.imshow(eroded)
plt.imshow(img, alpha=0.5)
plt.show()

np.sum(filled)*(actimg.resolution['pixel_size_xy']**2)//1e6

