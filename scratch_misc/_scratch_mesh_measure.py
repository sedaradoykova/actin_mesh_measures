"""  We then applied a threshold at an arbitrary low value (0.002), 
    resulting in a binary image of the actin meshwork (fig. S3).
    To establish the total area of the cell and of the mesh, 
    we “filled” the binary image so that any gaps in the network were removed, 
    and the cell was segmented throughout the whole plane. 
    The number of pixels in the “unfilled” image and the filled image was then 
    compared to give a percentage value of mesh density. 
    To establish the size of the gaps in the mesh, the unfilled image was inverted, 
    and the gaps within the mesh were characterized using the MATLAB “region props” function 
    including the “EquivDiameter” and “Area” parameters. 
    These parameters generated a circular area of equivalent size to the gap region 
    and then determined the meshwork size of that specific gap as the diameter of the circle. 
"""

import os
import numpy as np 
import matplotlib.pyplot as plt
from meshure.actinimg import get_ActinImg
from scipy.ndimage import correlate, morphology, label, binary_closing
from skimage import measure
from skimage import morphology as skimorph


data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/sample_data/sample_images_threshold")
all_files = [f for f in os.listdir(data_path) if 'tif' in f]
# Untr 3min_Fri_FOV2_decon.tif ## 1,3 and 6,8
# CARs 3min_FOV3_decon.tif ## 1 and ??? 
planes = [1,2], [1], [3,5], [1,3], [1,3], [3,4], [1,4], [3,4]
#[n for n, f, in enumerate(all_files) if f=='8min_Thurs_FOV2_decon.tif']



### TRY DIFFERENT MORPHOLOGICAL OPERATIONS 
# it appears that closing is indeed the best operation
# the problem remains: what structure to choose.... 


actimg = get_ActinImg('8min_Thurs_FOV2_decon.tif', data_path) 
actimg.normalise()
# actimg.z_project_max([3,4])
# im_og = actimg.manipulated_stack.copy()

#actimg.manipulated_stack = gaussian(actimg.manipulated_stack, 2)    # try this!!!! 
actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0,180,20),sigma=2,substack=[3,4],visualise=False)
actimg.z_project_min()

mu, sigma = actimg.threshold_dynamic(sigma_factor=0, return_mu_sigma=True)
actimg.visualise('manipulated', colmap='gray')


img = np.copy(actimg.manipulated_stack)
img_inverted = (img==0).astype('int')

# closing 
# allows to specify a (square?) structure which is used to close the holes in the image 
# closing = dilation followed by erosion with the same structuring element 
closed_image = binary_closing(img, structure = np.ones((15,15))) # compare    5,5   7,7   10,10   15,15   30,30
difference_closed = ~closed_image+img
labeled_image, num_features = label(closed_image)

# filling 
#filled_image = morphology.binary_fill_holes(img)
filled_image = morphology.binary_fill_holes(img)#,structure=np.ones((5,5)))
difference_filled = ~filled_image+img
##### CHANGE OF SIGNS DUE TO INVERTED THRESHOLDING!! 

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


# area opening 
# open inverted image and display inverted opened image? 
opened_image = skimorph.area_opening(img, 100) # area_threshold, manual adjust 
difference = ~opened_image+img
# opening thins the mesh so the inverse expression produces visible output i.e. ~img+opened_image 

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('binary image')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(opened_image, cmap='gray')
plt.title('opened image')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(difference, cmap='gray')
plt.title('difference image')
plt.axis('off')
plt.show();



# mesh density

mesh_density = ( np.sum(filled_image) - np.sum(img) )*100 / np.sum(filled_image)
print(f'The percentage mesh density is  {mesh_density:.2f} %')
print('Defined as the difference between the filled and unfilled mask.')



#labels = measure.label(img, background=1) # same as inverted when background set
labels = measure.label(img_inverted)
#props = measure.regionprops(labels, img)
props = measure.regionprops(labels)

for prop in props:
    print('Label: {} >> Object size: {}'.format(prop.label, prop.area))


#plt.imshow(labels, cmap=plt.cm.gnuplot); plt.show();

plt.imshow(labels, cmap='tab20c_r'); plt.show();


from scipy.ndimage import label
labeled_image, num_features = label((img == 0).astype('int'))
feature_areas = np.bincount(labeled_image.ravel())[1:]   
print(feature_areas) 

plt.imshow(labeled_image, cmap='inferno'); plt.show();



plt.subplot(1,2,1)
plt.imshow(labels, cmap='nipy_spectral')
plt.axis('off')
plt.title('measure.label')
plt.subplot(1,2,2)
plt.imshow(labeled_image, cmap='tab20c_r')
plt.axis('off')
plt.title('scipy.ndimage.label')
plt.show()



### show result

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title('binary image')
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(filled_image, cmap='gray')
plt.title('filled image')
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
plt.show();


diams = [prop.equivalent_diameter for prop in props]
plt.hist(diams, 200)
#plt.xlim(0, 200)
plt.show()

from skimage import morphology as skimorph
plt.imshow(morphology.binary_fill_holes(skimorph.binary_closing(img, skimorph.square(2)))); plt.show()

plt.imshow(morphology.binary_fill_holes(binary_closing(img,structure=np.ones((2,2))))); plt.show()

# could be closing followed by filling 

