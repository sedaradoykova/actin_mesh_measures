import os
import numpy as np 
import matplotlib.pyplot as plt
from meshure.actimg import get_ActImg


data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/sample_data/sample_images_threshold")
all_files = [f for f in os.listdir(data_path) if 'tif' in f]
# Untr 3min_Fri_FOV2_decon.tif ## 1,3 and 6,8
# CARs 3min_FOV3_decon.tif ## 1 and ??? 
planes = [1,2], [1], [3,5], [1,3], [1,3], [3,4], [1,4], [3,4]
#[n for n, f, in enumerate(all_files) if f=='8min_Thurs_FOV2_decon.tif']



for (im_title, subs) in zip(all_files, planes):
    actimg = get_ActImg(im_title, data_path) 
    actimg.normalise()
    actimg.z_project_max(substack=subs)
    actimg.visualise('manipulated', colmap='gray', save=True, dest_dir=data_path)
    actimg.nuke()
    actimg.normalise()
    actimg.steerable_gauss_2order_thetas(thetas=[0,60,120],sigma=2,substack=subs,visualise=False)
    actimg.z_project_min()
    rows, cols = actimg.shape
    lineprof_coords = [[(0,0),(rows,cols)], [(rows,0),(0,cols)], 
                    [(int(rows/2),0),(int(rows/2),cols)], [(0,int(cols/2)),(rows,int(cols/2))]]
    mu, sigma = actimg.threshold_find_mu_sigma(line_prof_coords=lineprof_coords)



    actimg.threshold(mu-sigma)
    actimg.visualise('manipulated', colmap='gray', save=True, dest_dir=data_path)





for (im_title, subs) in zip(os.listdir(data_path), planes):
    actimg = get_ActImg(im_title, data_path) 
    actimg.normalise()
    actimg.z_project_max(substack=subs)
    #actimg.steerable_gauss_2order_thetas(thetas=[0,60,120],sigma=2,visualise=False)
    rows, cols = actimg.shape
    lineprof_coords = [[(0,0),(rows,cols)], [(rows,0),(0,cols)], 
                    [(int(rows/2),0),(int(rows/2),cols)], [(0,int(cols/2)),(rows,int(cols/2))]]
    mu, sigma = actimg.threshold_find_mu_sigma(line_prof_coords=lineprof_coords)



    #actimg.threshold(mu-sigma)
    actimg.manipulated_stack = actimg.manipulated_stack > mu+2*sigma
    actimg.visualise('manipulated', colmap='gray', save=True, dest_dir=data_path)



from skimage.filters import gaussian
## threshold raw image doesn't work 
actimg = get_ActImg('8min_Thurs_FOV2_decon.tif', data_path) 
actimg.normalise()
# actimg.z_project_max([3,4])
# im_og = actimg.manipulated_stack.copy()

#actimg.manipulated_stack = gaussian(actimg.manipulated_stack, 2)    # try this!!!! 
actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0,180,20),sigma=2,substack=[3,4],visualise=False)
actimg.z_project_min()

mu, sigma = actimg.threshold_dynamic(sigma_factor=0, return_mu_sigma=True)
actimg.visualise('manipulated', colmap='gray')

# compare gaussian smooth vs no smooth

smoothed = actimg.manipulated_stack.copy()
not_sm = actimg.manipulated_stack.copy()

vmax, vmin = np.max([not_sm, smoothed]), np.min([not_sm, smoothed])
plt.subplot(2,2,1)
plt.imshow(not_sm,cmap='gray')#, vmax=vmax, vmin=vmin)
plt.title('not smoothed'); plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(smoothed,cmap='gray')#, vmax=vmax, vmin=vmin)
plt.title('smoothed'); plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(smoothed-not_sm, cmap='gray')
plt.title('smooth-not_smooth'); plt.axis('off')
plt.subplot(2,2,4)
plt.imshow(not_sm-smoothed, cmap='gray')
plt.title('not_smooth-smooth'); plt.axis('off')
plt.tight_layout()
plt.show()


actimg.manipulated_stack = not_sm-smoothed
actimg.manipulated_stack = smoothed-not_sm
actimg._threshold_preview_cases(mu, sigma, factors=[0,-0.25,-0.5],max_proj_substack=[3,4], save=False)
mu, sigma = actimg.threshold_dynamic(sigma_factor=0, return_mu_sigma=True)
actimg.visualise('manipulated', colmap='gray')


## preview cases

actimg.nuke()
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0, 180, 20),sigma=2,substack=[3,4],visualise=False)
actimg.z_project_min()
actimg._threshold_preview_cases(mu, sigma, factors=[0,0.25,0.5,1,1.25],max_proj_substack=[3,4], save=True)


# plt.imshow(im_og, cmap='inferno')
# plt.imshow(actimg.manipulated_stack, alpha=0.25)
# plt.show()

plt.imshow(actimg.manipulated_stack, cmap='gray')
plt.imshow(im_og, cmap='inferno', alpha=0.7)
plt.axis('off')
plt.show()


### make gif of oriented filters 
actimg.nuke()
actimg.normalise()
oriented_filters = actimg._visualise_oriented_filters(thetas=np.linspace(0, 180, 30),sigma=2, return_filters=True)#,save=True, dest_dir='misc_orient_filter')

flts = np.asarray(list(oriented_filters.values()))
flts = np.array([np.uint8((im - np.min(im))*5000) for im in flts])
np.max(flts[0])
from PIL import Image


imgs = [Image.fromarray(img, mode='L') for img in flts]
img = imgs[0].copy() # extract first image from iterator
img.save(fp="orient_filts.gif", format='GIF', append_images=imgs[1:],
         save_all=True, duration=200, loop=0)


# make gif preview filter responses 
actimg.nuke()
actimg.normalise()
flt_resp = actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0, 180, 31),sigma=2, substack=[3,4], visualise=False, return_responses=True)
flt_resp = np.min(flt_resp, 1)

flts = np.array([np.uint8((im - np.min(im))*1000) for im in flt_resp])

from PIL import Image

imgs = [Image.fromarray(img, mode='L') for img in flts]
img = imgs[0].copy() # extract first image from iterator
img.save(fp="orient_reps.gif", format='GIF', append_images=imgs[1:],
         save_all=True, duration=200, loop=0)



## using flt_resp above to color code by max angle response 
min_locs = np.argmin(flt_resp, 0)
for num_in, num_out in zip(np.unique(min_locs), np.linspace(0, 180, 31)):
    min_locs[np.where(min_locs == num_in)] = num_out

# overlay with response
actimg = get_ActImg('8min_Thurs_FOV2_decon.tif', data_path) 
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0, 180, 31),sigma=2,substack=[3,4],visualise=False)
actimg.z_project_min()
response = actimg.manipulated_stack.copy()

plt.imshow(response, cmap='gray')
plt.imshow(min_locs, cmap='hsv', alpha=0.3); 
plt.axis('off'); plt.colorbar(); plt.tight_layout(); plt.show()



## compare effect of sampling thetas on response 
actimg.nuke()
actimg.normalise()
flt_resp = actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0, 180, 61),sigma=2, substack=[3,4], visualise=False, return_responses=True)
flt_resp = np.min(flt_resp, 1)

for n,num in enumerate([1,2,5,10,20]):
    imin = flt_resp[::num,300:545,170:550]
    imout = np.min(imin, 0)
    plt.subplot(2,5,n+6)
    plt.imshow(imout)
    plt.axis('off')
    imout = imout < np.mean(imout)
    plt.subplot(2,5,n+1)
    plt.imshow(imout)
    plt.axis('off')
    plt.title(f'{imin.shape[0]} thetas used')
plt.tight_layout()
plt.show()


plt.imshow(np.min(flt_resp[::1,300:545,170:550], 0) - np.min(flt_resp[::20,300:545,170:550], 0)); plt.show()

plt.imshow((np.min(flt_resp[::1,:,:], 0) < np.mean(np.min(flt_resp[::1,:,:], 0))).astype('int') - (np.min(flt_resp[::20,:,:], 0) < np.mean(np.min(flt_resp[::20,:,:], 0))).astype('int')); plt.show()



### image + lines + aggregated profile histogram + response histogram 

from skimage.measure import profile_line

actimg = get_ActImg('8min_Thurs_FOV2_decon.tif', data_path) 
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0,180,20),sigma=2,substack=[3,4],visualise=False)
actimg.z_project_min()

rows, cols = actimg.shape
lineprof_coords = [[(0,0),(rows,cols)], [(rows,0),(0,cols)], 
                   [(int(rows/2),0),(int(rows/2),cols)], [(0,int(cols/2)),(rows,int(cols/2))]]
line_profs = [] 
for start, end in lineprof_coords: 
    line_profs.append(profile_line(actimg.manipulated_stack, start, end, linewidth=5).ravel())

all_profs = np.concatenate(line_profs).ravel()
all_profs.sort()

mu, sigma = np.mean(all_profs), np.std(all_profs)

plt.subplot(1,2,1)
plt.imshow(actimg.manipulated_stack, cmap='gray')
for (r1,r2),(c1,c2) in lineprof_coords: 
    plt.plot((r1,c1),(r2,c2), color='black')
plt.axis('off')
plt.title('Minimum projection of steerable filter resposnes')
plt.subplot(2,2,2)
ns, bins, _ = plt.hist(all_profs, 100, density=1, color='#A8A8A8') # alpha=0.5
binned_prof = (bins+(bins[1] - bins[0])/2)[:-1]
mu, sigma = np.mean(all_profs), np.std(all_profs)
plt.vlines(x = [mu, mu-sigma, mu+sigma], ymin=0, ymax=np.max(ns), 
           linestyles=['-','--','--'], colors=['black','gray', 'gray'], 
           label=f'mean({mu:.4f})+-sigma({sigma:.4f})')
plt.ylabel('count')
plt.title('Aggregated line intensity profiles')
plt.legend(loc='upper left')
plt.subplot(2,2,4)
ns, bins, _ = plt.hist(actimg.manipulated_stack.ravel(), 100, density=1, color='#A8A8A8')
mu, sigma = np.mean(actimg.manipulated_stack.ravel()), np.std(actimg.manipulated_stack.ravel())
plt.vlines(x = [mu, mu-sigma, mu+sigma], ymin=0, ymax=np.max(ns), 
           linestyles=['-','--','--'], colors=['black','gray', 'gray'], 
           label=f'mean({mu:.4f})+-sigma({sigma:.4f})')
plt.legend(loc='upper left')
plt.xlabel('intensity'); plt.ylabel('count')
plt.title('Response histogram')
plt.show()




### morphoogical operations 

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
from meshure.actimg import get_ActImg
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

actimg = get_ActImg('8min_Thurs_FOV2_decon.tif', data_path) 
actimg.normalise()
# actimg.z_project_max([3,4])
# im_og = actimg.manipulated_stack.copy()

#actimg.manipulated_stack = gaussian(actimg.manipulated_stack, 2)    # try this!!!! 
actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0,180,20),sigma=2,substack=[3,4],visualise=False)
actimg.z_project_min()

mu, sigma = actimg.threshold_dynamic(sigma_factor=0, return_mu_sigma=True)
actimg.visualise('manipulated', colmap='gray')


img = np.copy(actimg.manipulated_stack)
#img = np.copy(thresh)

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
