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


"""normalise --> steer 2o Gauss --> min z proj --> threshold"""
theta_x6 = np.arange(0,360,60)

actimg = get_ActinImg(filename, os.path.join(data_path,celltype))

actimg.visualise_stack(imtype='original',save=True,dest_dir=save_destdir)

actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=theta_x6,sigma=2,substack=[1,3],visualise=False)
#actimg._visualise_oriented_filters(thetas=theta_x6,sigma=2,save=True,dest_dir=save_destdir)

actimg.visualise_stack(imtype='manipulated',save=True,dest_dir=save_destdir)

actimg.z_project_min()
actimg.visualise_stack(imtype='manipulated',save=True,dest_dir=save_destdir)

actimg.threshold(0.002)
actimg.visualise_stack(imtype='manipulated',save=True,dest_dir=save_destdir)




actimg.normalise()
actimg.visualise_stack(imtype='manipulated', substack=[11,14])
res = actimg.steerable_gauss_2order([1,3],tmp=True)
actimg.visualise_stack(imtype='manipulated', substack=[1,3])

actimg = get_ActinImg(filename, os.path.join(data_path,celltype))
actimg.normalise()
actimg.z_project_max()
actimg.visualise('manipulated')

actimg.nuke()
actimg.normalise()

actimg.visualise_stack(imtype='original', substack=[1,3])
actimg.visualise_stack(imtype='manipulated')



actimg.visualise('manipulated')
actimg.threshold(0.03)
actimg.visualise('manipulated')


actimg = get_ActinImg(filename, os.path.join(data_path,celltype))
actimg.normalise()
#actimg._visualise_oriented_filters(np.arange(0,360,60),2)
actimg.steerable_gauss_2order_thetas(np.arange(0,360,60),2,[1,3], visualise=False)
actimg.z_project_min()
actimg.threshold(0.003)
actimg.visualise('manipulated')
