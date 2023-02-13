import os, time, subprocess, shutil, warnings
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
from actin_meshwork_analysis.meshwork.actinimg import get_ActinImg
from actin_meshwork_analysis.meshwork.utils import list_files_dir_str, search_files_root

""" TODO: 
    - add checks to make sure everything is parametrised 
    - run to see if it works with sample_data
    - make pipeline customisable 
    - document each function
"""

@dataclass()
class ActinImgCollection:
    root_path: str
    res_path: str=None
    only_subdirs: list[str]=None
    specify_subdirs: list=None
    focal_planes: pd.DataFrame=None
    analysis_steps=None
    parameters=None


    def __post_init__(self):
        if not isinstance(self.root_path, str):
            raise TypeError('Root path must be a string.')
        if not os.path.exists(self.root_path):
            raise ValueError(f'Path not found: {self.root_path}.')
        if self.res_path is not None and not isinstance(self.res_path, str):
            raise TypeError('Results path must be a string.')
        if self.res_path is not None and not os.path.exists(self.res_path):
            raise ValueError(f'Results path not found: {self.root_path}.')
        # read in data and extract file names/paths.
        self._all_filenames, self._all_filepaths = list_files_dir_str(self.root_path)
        self._filenames_to_del = None



    def print_summary(self, specify_subdirs=None, interact=False):
        """ Summary of files. """
        if not isinstance(interact, bool): 
            raise TypeError('interact must be a boolean.')

        if interact: 
            inp = input(f'Subdirs not specified. Print summary for all subdirs in {os.path.basename(self.root_path)}? [y/n]\n')
            if inp.lower() == 'n':
                raise InterruptedError('')
            elif inp.lower() == 'y':
                pass
            else: 
                raise ValueError('Invalid input, rerun command.')

        if specify_subdirs:
            if not isinstance(specify_subdirs, list): 
                raise TypeError('subdirs must be a list.')
            if any([not isinstance(item, str) for item in specify_subdirs]):
                raise TypeError('subdirs list must contain only strings.')
        
        for cat in ['1min', '3min', '8min']:
            for key, vals in self._all_filenames.items():
                if specify_subdirs and any(key == str(keys) for keys in specify_subdirs):
                    print(key, " "*(50-len(key)), ":", cat, " :", len([val for val in vals if cat in val]))
                elif specify_subdirs is None:
                    print(key, " "*(50-len(key)), ":", cat, " :", len([val for val in vals if cat in val]))
        print("")

        total_len = []
        for key, vals in self._all_filenames.items():
            if specify_subdirs is not None and any(key == str(keys) for keys in specify_subdirs):
                print(key, " "*(50-len(key)), ": total :", len(vals))
                total_len.append(len(vals))
            elif specify_subdirs is None:            
                print(key, " "*(50-len(key)), ": total :", len(vals))
                total_len.append(len(vals))
        
        print("\nTotal files", " "*39, ":", sum(total_len))



    def get_focal_planes(self, filepath):
        """ Read in focal planes. 
        
        Arguments
        ---------
        filepath : str 
            Filepath specifies a csv file with four fields (named exactly: 'File name', 'Type', 'Basal', 'Cytoplasmic', 'Notes').
                File name : name of tiff file 
                Type : cell type, either 'Untransduced' or 'CAR'
                Basal : one or two comma separated integers specifying the focal plane (or focal plane [start, end])
                    These are converted to a list inside the function
                Cytoplasmic : one or two comma separated integers specifying the focal plane (or focal plane [start, end])
                    These are converted to a list inside the function
                Notes : field not used by this function. 
        """
        if not isinstance(filepath, str):
            raise TypeError('filepath path must be a string.')
        if not os.path.exists(filepath):
            raise ValueError(f'Path not found: {filepath}.')

        focal_planes = pd.read_csv(filepath)
        # if list(focal_planes.columns) != ['File name', 'Basal', 'Cytoplasmic', 'Notes']:
        #     raise ValueError("Csv file contains unexpected columns. Columns must be named ['File name', 'Type', 'Basal', 'Cytoplasmic', 'Notes']")

        # if basal or cytosolic planes are not speficied, discard   
        self._filenames_to_del = focal_planes['File name'][focal_planes[['Basal', 'Cytoplasmic']].isna().any(axis=1)].tolist()
        focal_planes = focal_planes.dropna(subset=['Basal', 'Cytoplasmic'])

        focal_planes['Basal'] = focal_planes['Basal'].apply(lambda val: [int(x) for x in val.split(',')])
        focal_planes['Cytoplasmic'] = focal_planes['Cytoplasmic'].apply(lambda val: [int(x) for x in val.split(',')])

        self.focal_planes=focal_planes

        
    def initialise_curr_filenames_and_planes(self, subdir):
        """ Called inside analysis loop. Subsets the current filenames which are being processed based on subdir. 
        
        Arguments
        ---------
            sundir : str 
                ?????? 
        """
        curr_filenames, curr_filepath = self._all_filenames[subdir], self._all_filepaths[subdir]
        curr_filenames = [f for f in curr_filenames if 'tif' in f and f not in self._filenames_to_del]
        if 'untr' in subdir.lower():
            curr_planes = self.focal_planes[self.focal_planes['Type'] == 'Untransduced']
        elif 'car' in subdir.lower():
            curr_planes = self.focal_planes[self.focal_planes['Type'] == 'CAR']
        else:
            raise ValueError('Cell type not recognised as either Untransduced or CAR.')

        if not any(curr_planes['File name'].sort_values() == sorted(curr_filenames)):
            raise ValueError(f'File names do not match between focal planes csv file and filenames listed in {curr_filepath}.')
        else: 
            return curr_filenames, curr_filepath


    def initialise_res_dir(self, subdir):
        """ Initialise results directory. """
        self.__save_destdir = self.res_path if self.res_path else os.path.join(self.root_path, '_results_'+subdir[0:4])
        if not os.path.exists(self.__save_destdir):
            os.mkdir(self.__save_destdir)

        self.__basal_dest = os.path.join(self.__save_destdir, 'basal')
        if not os.path.exists(self.__basal_dest):
            os.mkdir(self.__basal_dest)

        self.__cyto_dest = os.path.join(self.__save_destdir, 'cytosolic')
        if not os.path.exists(self.__cyto_dest):
            os.mkdir(self.__cyto_dest)

        self.__main_dest = os.path.join(self.__save_destdir, 'main')
        if not os.path.exists(self.__main_dest):
            os.mkdir(self.__main_dest)

    def define_pipeline(self):
        # which functions to call 
        # which intermediate results to save 
        raise NotImplementedError

    def parametrise_pipeline(self, substack=None, theta=None, thetas=None, sigma=2, threshold=0.002):
        """ Fill in parameters based on function calls in pipeline. 

        Arguments
        ---------
        substack : list of ints
        theta : float
        thetas : list of floats
        sigma : float
        threshold : float

        Returns 
        -------
        dict : ????
        """
        self.parameters = dict.fromkeys(['substack', 'theta', 'thetas', 'sigma', 'threshold'])
        for key, value in zip(self.parameters.keys(), [substack, theta, thetas, sigma, threshold]):
            self.parameters[key] = value



    def analysis_pipeline(self, filename, filepath, print_summary=False):
        """ ??? """
        actin_img_instance = get_ActinImg(filename, filepath)
        actin_img_instance.visualise_stack(imtype='original',save=True,dest_dir=self.__main_dest) 
        actin_img_instance.z_project_max()
        actin_img_instance.visualise_stack(imtype='manipulated',save=True,dest_dir=self.__main_dest)

        basal_stack, cyto_stack = self.focal_planes.loc[self.focal_planes['File name']==filename, ['Basal', 'Cytoplasmic']].values[0]

        for stack, dest in zip([basal_stack, cyto_stack], [self.__basal_dest, self.__cyto_dest]):
            # !!!!! INTEGRATE PIPELINE HERE 
            # fix parametrisation!!!! 
            """ Analysis:
                    max z proj on raw data
                    nuke
                    basal: normalise --> steer 2o Gauss --> min z proj --> threshold
                    nuke 
                    cytosolic: normalise --> steer 2o Gauss --> min z proj --> threshold 
            """
            actin_img_instance.nuke()
            actin_img_instance.normalise()

            actin_img_instance.steerable_gauss_2order_thetas(thetas=self.parameters['thetas'],sigma=self.parameters['sigma'],substack=stack,visualise=False)
            #actimg._visualise_oriented_filters(thetas=theta_x6,sigma=2,save=True,dest_dir=save_destdir)
            actin_img_instance.visualise_stack(imtype='manipulated',save=True,dest_dir=dest)

            actin_img_instance.z_project_min()
            actin_img_instance.visualise_stack(imtype='manipulated',save=True,dest_dir=dest)

            actin_img_instance.threshold(self.parameters['threshold'])
            actin_img_instance.visualise_stack(imtype='manipulated',save=True,dest_dir=dest)

        if print_summary:
            self.print_summary([os.path.basename(self.__save_destdir)]) 


    def visualise_html(self, subdir):
        """ ??? 
        Returns
        -------
        hidden: makes file named as ..... 
        """
        curr_filenames, curr_filepath = self._all_filenames[subdir], self._all_filepaths[subdir]
        curr_filenames = [f for f in curr_filenames if 'tif' in f and f not in self._filenames_to_del]

        all_output_types = dict.fromkeys(['original', 'max_proj', 'steer_gauss', 'min_proj', 'threshold'])

        out_types = ['original', 'max_proj']
        type_file_ends = ['original', 'max']
        results_filenames = [res for res in os.listdir(self.__save_destdir+'/main') if 'png' in res]
        for key, val_check in zip(out_types, type_file_ends):
            all_output_types[key] = [res for res in results_filenames if val_check in res]


        out_types = ['steer_gauss', 'min_proj', 'threshold']
        type_file_ends = ['300+6.png', 'min.png', 'threshold.png']
        results_filenames = [res for res in os.listdir(self.__save_destdir+'/basal') if 'png' in res]
        for key, val_check in zip(out_types, type_file_ends):
            all_output_types[key] = [res for res in results_filenames if val_check in res]

        md_filename = subdir[0:10]+'.md'
        results_base_dir = os.path.basename(self.__save_destdir)
        with open(os.path.join(self.__save_destdir,md_filename), 'w') as f:
            for i in range(len(curr_filenames)):
                f.write(f'# {curr_filenames[i]}')
                f.write('\n\n')
                f.write(f'![](main/{all_output_types["original"][i]})'+'{ width=700px } ')
                f.write('\n\n')
                f.write('**Maximum z-projection**  ')
                f.write('\n')
                f.write(f'![](main/{all_output_types["max_proj"][i]})'+'{ height=300px }  ')
                f.write('\n\n')
                f.write('## Basal network  ')
                f.write('\n\n')
                f.write('**Steerable second order Gaussian filter**  ')
                f.write('\n')
                f.write(f'![](basal/{all_output_types["steer_gauss"][i]})'+'{ height=300px }  ')
                f.write('\n')
                f.write('**Minimum z-projection**  ')
                f.write('\n')
                f.write(f'![](basal/{all_output_types["min_proj"][i]})'+'{ height=300px }  ')
                f.write('\n')
                f.write('**Binary thresholding**  ')
                f.write('\n')
                f.write(f'![](basal/{all_output_types["threshold"][i]})'+'{ height=300px }  ')
                f.write('\n\n')
                f.write('## Cytosolic network  ')
                f.write('\n\n')
                f.write('**Steerable second order Gaussian filter**  ')
                f.write('\n')
                f.write(f'![](cytosolic/{all_output_types["steer_gauss"][i]})'+'{ height=300px }  ')
                f.write('\n')
                f.write('**Minimum z-projection**  ')
                f.write('\n')
                f.write(f'![](cytosolic/{all_output_types["min_proj"][i]})'+'{ height=300px }  ')
                f.write('\n')
                f.write('**Binary thresholding**  ')
                f.write('\n')
                f.write(f'![](cytosolic/{all_output_types["threshold"][i]})'+'{ height=300px }  ')
                f.write('\n\n\n')
            

        pandoc_input = os.path.join(self.root_path, results_base_dir, md_filename)
        if shutil.which('pandoc') is None:
            warnings.warn(f'Pandoc is not installed. Returning only {pandoc_input.replace(".md", ".html")}.')
        else: 
            subprocess.run(f'pandoc -t slidy -s {pandoc_input} -o {pandoc_input.replace(".md", ".html")}', shell=True)


    def run_analysis(self, visualise_as_html=False):
        """ Runs analysis on selected subdirs. """
        if self.parameters is None: 
            raise AttributeError('self.parameters has not been initialised; call self.parametrise_pipeline().')

        t_start = time.time()
        for subdir in tqdm(self.only_subdirs, desc='cell types'):

            curr_filenames, curr_filepath = self.initialise_curr_filenames_and_planes(subdir=subdir)
            self.initialise_res_dir(subdir=subdir)

            for name in tqdm(curr_filenames, desc='files'):
                self.analysis_pipeline(filename=name,filepath=curr_filepath, print_summary=True)
        
            if visualise_as_html:
                self.visualise_html(subdir=subdir)

        delta_t = time.time() - t_start
        print(f'Analysis completed in {time.strftime("%H:%M:%S", time.gmtime(delta_t))}.')
        
    def next(self):
        n=0
        raise NotImplementedError





if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/sample_data")
    #data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/deconv_data")


    only_subdirs = ['Untransduced_1.11.22_processed_imageJ','CARs_8.11.22_processed_imageJ']
    focal_plane_filename = os.path.join(data_path, 'basal_cytosolic_focal_planes_v2.csv')

    theta_x6 = np.arange(0,360,60)


    sample_data = ActinImgCollection(root_path=data_path)

"""
get_ActinImg(filename, filepath)
visualise_stack(imtype='original',save=True,dest_dir=self.__main_dest) 
z_project_max()
visualise_stack(imtype='manipulated',save=True,dest_dir=self.__main_dest)

nuke()
normalise()

steerable_gauss_2order_thetas(thetas=self.parameters['thetas'],sigma=self.parameters['sigma'],substack=stack,visualise=False)
#_visualise_oriented_filters(thetas=theta_x6,sigma=2,save=True,dest_dir=save_destdir)
isualise_stack(imtype='manipulated',save=True,dest_dir=dest)

z_project_min()
visualise_stack(imtype='manipulated',save=True,dest_dir=dest)

threshold(self.params['threshold'])
visualise_stack(imtype='manipulated',save=True,dest_dir=dest)

r = {'initialise': {'vis': True, 'params': ["imtype='original',save=True,dest_dir=self.__main_dest"]}}
r['initialise']['vis']
{'z_proj_max', {'vis': True,}}
steerable_gauss_thetas, vis=True
"""