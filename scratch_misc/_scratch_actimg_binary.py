import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from meshure.actimg import get_ActImg
from meshure.actimg_binary import ActImgBinary, get_ActImgBinary
data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/deconv_data/")

#3min_FOV4_decon_top_right
subs = [3,4]
actimg = get_ActImg('1min_FOV8_decon_bottom_left.tif ', os.path.join(data_path, 'CARs_8.11.22_processed_imageJ')) 
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0,180,20),sigma=2,substack=subs,visualise=False)
actimg.z_project_min()
actimg.threshold_dynamic(std_dev_factor=0, return_mean_std_dev=False)

actimg.meshwork_density(False)
actimg.meshwork_size(False, False, False) #FFT
#actimg.surface_area(True)
#actimg.visualise_stack('manipulated')


new = get_ActImgBinary(actimg)
#new.visualise('original', 1)
new.surface_area(n_dilations_erosions=(0,2),closing_structure=None,extra_dilate_fill=True,verbose=True)
new.mesh_holes_area(visualise=True)
new.visualise_segmentation()
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
