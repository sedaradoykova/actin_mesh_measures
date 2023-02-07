import os
from actin_meshwork_analysis.meshwork.utils import list_files_dir_str
import numpy as np 
import matplotlib.pyplot as plt
from actin_meshwork_analysis.meshwork.actinimg import get_ActinImg


data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/sample_data")
os.path.exists(data_path)

all_filenames, all_filepaths = list_files_dir_str(data_path)
celltype = 'Untransduced'
filename, filepath = all_filenames[celltype][0], all_filepaths[celltype][0] 

save_destdir = os.path.join(data_path, '_results')
if not os.path.exists(save_destdir):
    os.mkdir(save_destdir)


"""normalise --> steer 2o Gauss --> min z proj --> threshold; also max z proj on raw data"""
theta_x6 = np.arange(0,360,60)

# substacks for focal planes; arbitrarily defined for now
substack_list = [[1,3],[1,3],[3,8],[1,4],[1,3],[3,6],[3,6]]


for name, subst in zip(all_filenames[celltype], substack_list):
    actimg = get_ActinImg(name, all_filepaths[celltype])
    actimg.visualise_stack(imtype='original',save=True,dest_dir=save_destdir) 


    actimg.normalise()
    actimg.steerable_gauss_2order_thetas(thetas=theta_x6,sigma=2,substack=subst,visualise=False)
    #actimg._visualise_oriented_filters(thetas=theta_x6,sigma=2,save=True,dest_dir=save_destdir)
    actimg.visualise_stack(imtype='manipulated',save=True,dest_dir=save_destdir)


    actimg.z_project_min()
    actimg.visualise_stack(imtype='manipulated',save=True,dest_dir=save_destdir)


    actimg.threshold(0.002)
    actimg.visualise_stack(imtype='manipulated',save=True,dest_dir=save_destdir)


    actimg.nuke()
    actimg.z_project_max()
    actimg.visualise_stack(imtype='manipulated',save=True,dest_dir=save_destdir)


# # the output of cv2 and scipy functions matches for averaged out thetas with normalised image input
# np.save('cv2.npy', actimg.manipulated_stack)
# cv2_norm_mat = np.load('actin_meshwork_analysis\\process_data\\sample_data\\compare_cv2_scipy\\cv2_normalised.npy')
# scipy_norm_mat = np.load('actin_meshwork_analysis\\process_data\\sample_data\\compare_cv2_scipy\\scipy_normalised.npy')

# np.allclose(cv2_norm_mat, scipy_norm_mat, atol=1e-15)