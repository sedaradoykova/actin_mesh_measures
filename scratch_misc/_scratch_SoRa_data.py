import os
import numpy as np
import matplotlib.pyplot as plt
from meshure.actimg import get_ActImg

### sample SoRa data 

imname, data_path = "WT_15min_cell1.tif", "actin_meshwork_analysis/lab_files/sora_analysis_attempt"
subs = [70,73]

actimg = get_ActImg(imname, data_path)
# i could not read in the meta data, so i am specifying it here for now 
actimg.resolution = {'unit' : 'nm', 'pixel_size_xy': 38.7, 'pixel_size_z': 1}
actimg.visualise_stack(substack=subs,scale_bar=False)

actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0,180,20),sigma=2,substack=subs,visualise=False)
actimg.visualise_stack('manipulated',scale_bar=False,colmap='gray')

actimg.z_project_min()
actimg.visualise_stack('manipulated',scale_bar=False,colmap='gray')

actimg.threshold_dynamic(std_dev_factor=0, return_mean_std_dev=False)
actimg.visualise_stack('manipulated',scale_bar=False,colmap='gray')



# get parameters 
actimg.meshwork_density(verbose=True)
actimg.meshwork_size(summary=True, verbose=True, visualise=True) #FFT
actimg.surface_area(verbose=True)






# color code response by angle color  
## using flt_resp above to color code by max angle response 
actimg.nuke()
actimg.normalise()
flt_resp = actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0, 180, 20),sigma=2, substack=subs, visualise=False, return_responses=True)
flt_resp = np.min(flt_resp, 1)

min_locs = np.argmin(flt_resp, 0)
for num_in, num_out in zip(np.unique(min_locs), np.linspace(0, 180, 20)):
    min_locs[np.where(min_locs == num_in)] = num_out



# overlay with response
actimg.nuke()
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0, 180, 20),sigma=2,substack=subs,visualise=False)
actimg.z_project_min()
response = actimg.manipulated_stack.copy()
actimg.threshold_dynamic(std_dev_factor=0, return_mean_std_dev=False)
binary_mask = actimg.manipulated_stack.copy()
inds = np.where(binary_mask == 1)


plt.imshow(response, cmap='gray')
plt.imshow(min_locs, cmap='hsv', alpha=0.5); 
plt.axis('off'); plt.colorbar(); plt.tight_layout(); plt.show()



# overlay with binary mask 

new_img = np.zeros(binary_mask.shape)
new_img[inds] =  min_locs[inds].copy()

plt.imshow(binary_mask, cmap='gray')
plt.imshow(new_img, cmap='hsv', alpha=0.5); 
plt.axis('off'); plt.colorbar(); plt.tight_layout(); plt.show()
