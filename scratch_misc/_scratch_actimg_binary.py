import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from meshure.actimg import get_ActImg
from meshure.actimg_binary import ActImgBinary, get_ActImgBinary
data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/deconv_data/")


subs = [1]
actimg = get_ActImg('8min_FOV7_decon_top_right.tif ', os.path.join(data_path, 'CARs_8.11.22_processed_imageJ')) 
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=np.linspace(0,180,20),sigma=2,substack=subs,visualise=False)
actimg.z_project_min()
actimg.threshold_dynamic(std_dev_factor=0, return_mean_std_dev=False)

actimg.meshwork_density(False)
actimg.meshwork_size(False, False, False) #FFT
#actimg.surface_area(True)
#actimg.visualise('manipulated')


new = get_ActImgBinary(actimg)
#new.visualise('original', 1)
new.surface_area(n_dilations_erosions=(1,3),closing_structure=None,extra_dilate_fill=True)
new.mesh_holes_area(visualise=True)
new.visualise_segmentation()
new.mesh_density()
new.quantify_mesh()
#new.estimated_parameters['mesh_holes']['hole_parameters'].hist(); plt.show()

new.save_estimated_parameters('actin_meshwork_analysis/scratch_misc')

not new.estimated_parameters['cell_surface_area']
not new.cell_surface_area
new.estimated_parameters["mesh_holes"]["unit"]


new.estimated_parameters
new.labels
hole_parameters = pd.DataFrame(measure.regionprops_table(new.labels, new.binary_mesh, properties=['equivalent_diameter_area', 'area']))

hole_parameters.equivalent_diameter_area.hist(); plt.show()
hole_parameters.hist(); plt.show()


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