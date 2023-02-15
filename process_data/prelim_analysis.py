import os, time
import numpy as np 
import pandas as pd
from tqdm import tqdm
from actin_meshwork_analysis.meshwork.actinimg import get_ActinImg
from actin_meshwork_analysis.meshwork.utils import list_files_dir_str, search_files_root


""" Read in data and extract file names/paths. """

#data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/sample_data")
data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/deconv_data")
os.path.exists(data_path)

all_filenames, all_filepaths = list_files_dir_str(data_path)
celltypes = ['Untransduced_1.11.22_processed_imageJ','CARs_8.11.22_processed_imageJ']


""" Summary of files. """

def print_summary(all_filenames, specific_keys=None):
    for cat in ['1min', '3min', '8min']:
        for key, vals in all_filenames.items():
            if specific_keys and any(key == str(keys) for keys in specific_keys):
                print(key, " "*(50-len(key)), ":", cat, " :", len([val for val in vals if cat in val]))
            elif specific_keys is None:
                print(key, " "*(50-len(key)), ":", cat, " :", len([val for val in vals if cat in val]))
    print("")
    total_len = []
    for key, vals in all_filenames.items():
        if specific_keys is not None and any(key == str(keys) for keys in specific_keys):
            print(key, " "*(50-len(key)), ": total :", len(vals))
            total_len.append(len(vals))
        elif specific_keys is None:            
            print(key, " "*(50-len(key)), ": total :", len(vals))
            total_len.append(len(vals))


    print("\nTotal files", " "*39, ":", sum(total_len))

if __name__ == '__main__':
    print_summary(all_filenames, ['Untransduced_1.11.22_processed_imageJ', 'CARs_8.11.22_processed_imageJ'])

# CARs_8.11.22_processed_imageJ                       : 1min  : 16
# Untransduced_1.11.22_processed_imageJ               : 1min  : 11
# CARs_8.11.22_processed_imageJ                       : 3min  : 19
# Untransduced_1.11.22_processed_imageJ               : 3min  : 6
# CARs_8.11.22_processed_imageJ                       : 8min  : 17
# Untransduced_1.11.22_processed_imageJ               : 8min  : 5

# CARs_8.11.22_processed_imageJ                       : total : 53 --> 49 after processing
# Untransduced_1.11.22_processed_imageJ               : total : 22

# Total files                                         : 75


""" Read in focal planes. """

focal_planes = pd.read_csv('actin_meshwork_analysis\\process_data\\deconv_data\\basal_cytosolic_focal_planes_v2.csv')
to_del = focal_planes['File name'][focal_planes[['Basal', 'Cytoplasmic']].isna().any(axis=1)].tolist()
focal_planes = focal_planes.dropna(subset=['Basal', 'Cytoplasmic'])


focal_planes['Basal'] = focal_planes['Basal'].apply(lambda val: [int(x) for x in val.split(',')])
focal_planes['Cytoplasmic'] = focal_planes['Cytoplasmic'].apply(lambda val: [int(x) for x in val.split(',')])



""" Analysis:
        max z proj on raw data
        nuke
        basal: normalise --> steer 2o Gauss --> min z proj --> threshold
        nuke 
        cytosolic: normalise --> steer 2o Gauss --> min z proj --> threshold """

t_start = time.time()

for celltype in celltypes:
    curr_filenames, curr_filepaths = all_filenames[celltype], all_filepaths[celltype]
    curr_filenames = [f for f in curr_filenames if 'tif' in f and f not in to_del]
    if 'untr' in celltype.lower():
        curr_planes = focal_planes[focal_planes['Type'] == 'Untransduced']
    elif 'car' in celltype.lower():
        curr_planes = focal_planes[focal_planes['Type'] == 'CAR']
    else:
        raise ValueError('Cell type not recognised as either Untransduced or CAR.')

    if not any(curr_planes['File name'].sort_values() == sorted(curr_filenames)):
        raise ValueError('File names do not match between csv and list')


    """ Initialise results directory. """
    save_destdir = os.path.join(data_path, '_results_'+celltype[0:4])
    if not os.path.exists(save_destdir):
        os.mkdir(save_destdir)

    basal_dest = os.path.join(save_destdir, 'basal')
    if not os.path.exists(basal_dest):
        os.mkdir(basal_dest)

    cyto_dest = os.path.join(save_destdir, 'cytosolic')
    if not os.path.exists(cyto_dest):
        os.mkdir(cyto_dest)


    theta_x6 = np.arange(0,360,60)


    for name in tqdm(curr_filenames):
        actimg = get_ActinImg(name, curr_filepaths)
        actimg.visualise_stack(imtype='original',save=True,dest_dir=save_destdir) 
        actimg.z_project_max()
        actimg.visualise_stack(imtype='manipulated',save=True,dest_dir=save_destdir)

        basal_stack, cyto_stack = focal_planes.loc[focal_planes['File name']==name, ['Basal', 'Cytoplasmic']].values[0]

        for stack, dest in zip([basal_stack, cyto_stack], [basal_dest, cyto_dest]):
            actimg.nuke()
            actimg.normalise()

            actimg.steerable_gauss_2order_thetas(thetas=theta_x6,sigma=2,substack=stack,visualise=False)
            #actimg._visualise_oriented_filters(thetas=theta_x6,sigma=2,save=True,dest_dir=save_destdir)
            actimg.visualise_stack(imtype='manipulated',save=True,dest_dir=dest)

            actimg.z_project_min()
            actimg.visualise_stack(imtype='manipulated',save=True,dest_dir=dest)

            actimg.threshold(0.002)
            actimg.visualise_stack(imtype='manipulated',save=True,dest_dir=dest)
    
    """ Summarise preliminary results. """
    all_results, all_respaths = list_files_dir_str(os.path.join(data_path, save_destdir))
    print_summary(all_results)

delta_t = time.time() - t_start
print(f'Analysis completed in {time.strftime("%H:%M:%S", time.gmtime(delta_t))}.')

# Analysis completed in 00:13:00.

### UNTRANSDUCED 
# basal                                               : 1min  : 33 == 11*3
# cytosolic                                           : 1min  : 33 
# basal                                               : 3min  : 18 == 6*3
# cytosolic                                           : 3min  : 18
# basal                                               : 8min  : 15 == 5*3
# cytosolic                                           : 8min  : 15
# basal                                               : total : 66 == 22*3
# cytosolic                                           : total : 66

# Total files                                         : 132

### CAR
# basal                                               : 1min  : 39 == 13*3
# cytosolic                                           : 1min  : 39
# basal                                               : 3min  : 57 == 19*3
# cytosolic                                           : 3min  : 57
# basal                                               : 8min  : 51 == 17*3
# cytosolic                                           : 8min  : 51

# basal                                               : total : 147
# cytosolic                                           : total : 147

# Total files                                         : 294

all_cars = dict.fromkeys(['original', 'max_proj', 'steer_gauss', 'min_proj', 'threshold'])
carsres = [res for res in os.listdir(data_path+'/_results_CARs') if 'png' in res]

all_cars['original'] = [res for res in carsres if 'original' in res]
all_cars['max_proj'] = [res for res in carsres if 'max' in res]
carsres = [res for res in os.listdir(data_path+'/_results_CARs/basal') if 'png' in res]
all_cars['steer_gauss'] = [res for res in carsres if '300+6.png' in res]
all_cars['min_proj'] = [res for res in carsres if 'min.png' in res]
all_cars['threshold'] = [res for res in carsres if 'threshold.png' in res]

with open("actin_meshwork_analysis\\process_data\\deconv_data\\cars_all_prelim.md", "w") as f:
    for i in range(len(curr_filenames)):
        f.write(f'# {curr_filenames[i]}')
        f.write('\n\n')
        f.write(f'![](_results_CARs/{all_cars["original"][i]})'+'{ width=700px } ')
        f.write('\n\n')
        f.write('**Maximum z-projection**  ')
        f.write('\n')
        f.write(f'![](_results_CARs/{all_cars["max_proj"][i]})'+'{ height=300px }  ')
        f.write('\n\n')
        f.write('## Basal network  ')
        f.write('\n\n')
        f.write('**Steerable second order Gaussian filter**  ')
        f.write('\n')
        f.write(f'![](_results_CARs/basal/{all_cars["steer_gauss"][i]})'+'{ height=300px }  ')
        f.write('\n')
        f.write('**Minimum z-projection**  ')
        f.write('\n')
        f.write(f'![](_results_CARs/basal/{all_cars["min_proj"][i]})'+'{ height=300px }  ')
        f.write('\n')
        f.write('**Binary thresholding**  ')
        f.write('\n')
        f.write(f'![](_results_CARs/basal/{all_cars["threshold"][i]})'+'{ height=300px }  ')
        f.write('\n\n')
        f.write('## Cytosolic network  ')
        f.write('\n\n')
        f.write('**Steerable second order Gaussian filter**  ')
        f.write('\n')
        f.write(f'![](_results_CARs/cytosolic/{all_cars["steer_gauss"][i]})'+'{ height=300px }  ')
        f.write('\n')
        f.write('**Minimum z-projection**  ')
        f.write('\n')
        f.write(f'![](_results_CARs/cytosolic/{all_cars["min_proj"][i]})'+'{ height=300px }  ')
        f.write('\n')
        f.write('**Binary thresholding**  ')
        f.write('\n')
        f.write(f'![](_results_CARs/cytosolic/{all_cars["threshold"][i]})'+'{ height=300px }  ')
        f.write('\n\n\n')




# # the output of cv2 and scipy functions matches for averaged out thetas with normalised image input
# np.save('cv2.npy', actimg.manipulated_stack)
# cv2_norm_mat = np.load('actin_meshwork_analysis\\process_data\\sample_data\\compare_cv2_scipy\\cv2_normalised.npy')
# scipy_norm_mat = np.load('actin_meshwork_analysis\\process_data\\sample_data\\compare_cv2_scipy\\scipy_normalised.npy')

# np.allclose(cv2_norm_mat, scipy_norm_mat, atol=1e-15)
