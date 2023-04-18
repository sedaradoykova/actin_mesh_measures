import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from meshure.actimg import get_ActImg
from meshure.actimg_binary import ActImgBinary, get_ActImgBinary
data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/deconv_data/")


"""[
#    ['8min_UNT_FOV3_decon.tif', 'Untransduced', 'Cytosolic'],
#    ['8min_UNT_FOV6_decon.tif', 'Untransduced', 'Cytosolic'],
#    ['8min_UNT_FOV7_decon.tif', 'Untransduced', 'Cytosolic'],
# 8 min UNT = cropped out strange imaging artifact (lines like this < |_ >)
    ['3min_UNT_FOV1_decon.tif', 'Untransduced', 'Basal'],       # weird metadata 
    ['3min_UNT_FOV1_decon.tif', 'Untransduced', 'Cytosolic'],   # weird metadata 
    ['1min_FOV8_decon_right.tif', 'CAR', 'Cytosolic'],     # disconnected cell 
    ['3min_FOV1_decon_left.tif', 'CAR', 'Basal'],          # full surface cannot be segmented, poor signal 
    ['3min_FOV4_decon_top_right.tif', 'CAR', 'Cytosolic'], # disconnected cell is beyond saving 
#    ['3min_FOV5_decon_top.tif', 'CAR', 'Cytosolic'], 
    ['8min_FOV6_decon_top.tif','CAR', 'Cytosolic'],
    ['8min_FOV6_decon_top.tif','CAR', 'Basal']
]"""

#### MANUAL CORRECTIONS #### 

# 1min_FOV8_decon_right.tif; subs = [3,4]; new.contour_img[86:423,577] = 1
# surface area still not segmented more work needed 
# 3min_FOV1_decon_left new.surface_area(n_dilations_erosions=(5,0),closing_structure=None,extra_dilate_fill=True,verbose=False)
# surface area can't be patched, holes are not reliable 
subs = [5,7] #[1,3]
actimg = get_ActImg('8min_FOV6_decon_top.tif ', os.path.join(data_path, 'all_CARs')) 
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0,180,21),sigma=2,substack=subs,visualise=False)
actimg.z_project_min()
#actimg.manipulated_stack = actimg.manipulated_stack[:-20,50:]

actimg.threshold_dynamic(std_dev_factor=0, return_mean_std_dev=False)
actimg.visualise_stack(imtype='manipulated',substack=subs,save=False,colmap='gray')

new = get_ActImgBinary(actimg)
new.surface_area(n_dilations_erosions=(0,0),closing_structure=None,extra_dilate_fill=True,verbose=False)
print(new.log)
new.mesh_holes_area(visualise=True)
new.visualise_segmentation(save=False)
new.mesh_density()
new.quantify_mesh()
#new.estimated_parameters

plt.imshow(new.contour_img);plt.show()

new.contour_img[86:423,577] = 1

#### visualisations #### 

#3min_FOV4_decon_top_right
subs = [2,3] #[8,10]
actimg = get_ActImg('8min_CARs_Dual_FOV1_decon.tif ', os.path.join(data_path, 'all_CARs')) 
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0,180,20),sigma=2,substack=subs,visualise=False)
#actimg.manipulated_stack = actimg.manipulated_stack[0]
actimg.visualise_stack(imtype='manipulated',save=True,colmap='gray')
actimg.visualise_stack(imtype='original',substack=subs,save=True,colmap='gray')
actimg.visualise(imtype='original',ind=2,save=True,colmap='gray')

actimg.z_project_min()
#actimg.z_project_max(subs)
#actimg.visualise_stack('manipulated')
#actimg._threshold_preview_cases(factors=[+0.25, +0.5])
actimg.threshold_dynamic(std_dev_factor=0, return_mean_std_dev=False)

actimg.visualise_stack('manipulated')

new = get_ActImgBinary(actimg)
#new.visualise('original', 1)
new.surface_area(n_dilations_erosions=(0,2),closing_structure=None,extra_dilate_fill=True,verbose=False)
print(new.log)
#new.save_log()
new.mesh_holes_area(visualise=True)
new.visualise_segmentation(save=True)
new.mesh_density()
new.quantify_mesh()
#new.estimated_parameters['mesh_holes']['hole_parameters'].hist(); plt.show()

#new.save_estimated_parameters('actin_meshwork_analysis/scratch_misc')


plt.imshow(new.labels); plt.show()
new.cell_surface_area
new.f_labels_area
new.estimated_parameters["mesh_holes"]['hole_parameters']['area_um^2'].sum()
np.sum(new.mesh_outline)



# import h5py

# f = h5py.File('test.hdf5', 'w')
# for group_name in new.estimated_parameters:
#     group = f.create_group(group_name)
#     for data_name in new.estimated_parameters[group_name]:
#         dataset = group.create_dataset(data_name, data = new.estimated_parameters[group_name][data_name])
#         print(group_name, data_name, new.estimated_parameters[group_name][data_name])
# f.close()


plt.imshow(new.mesh_outline-new.eroded_contour_img); plt.show()
np.sum(new.mesh_outline)*100 / np.sum(new.eroded_contour_img)
np.unique(new.mesh_outline)

len('')
