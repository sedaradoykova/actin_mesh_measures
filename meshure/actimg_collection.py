import os, yaml, time, subprocess, shutil, warnings
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
from meshure.actimg import get_ActImg
from meshure.utils import list_files_dir_str, search_files_root

""" TODO: 
    - add checks to make sure everything is parametrised 
    - document each function
"""

@dataclass()
class ActImgCollection:
    root_path: str
    res_path: str=None
    only_subdirs: list[str]=None
    specify_subdirs: list=None
    focal_planes: pd.DataFrame=None
    analysis_steps=None
    parameters=None
    pipeline_outline = {# loop through these (if needed) for basal/cytosolic results 
            '01': {'func': 'normalise', 'params': None, 
                   'vis_stack': False, 'vis_params': "imtype='manipulated',save=True,dest_dir=self.parameters['dest_dir']"},
            '02': {'func': 'steerable_gauss_2order_thetas', 
                   'params': "thetas=self.parameters['thetas'],sigma=self.parameters['sigma'],substack=self.parameters['substack'],visualise=False",
                   'vis_stack': True, 'vis_params': "imtype='manipulated',save=True,dest_dir=self.parameters['dest_dir']"},
            '03': {'func': 'z_project_min', 'params': None,
                   'vis_stack': True, 'vis_params': "imtype='manipulated',save=True,dest_dir=self.parameters['dest_dir']"},
            '04': {'func': 'threshold_dynamic', 'params': "std_dev_factor=0,return_mean_std_dev=False",
                   'vis_stack': True, 'vis_params': "imtype='manipulated',save=True,dest_dir=self.parameters['dest_dir']"},
            '05': {'func': 'meshwork_density', 'params': "verbose=False", 
                   'vis_stack': False},
            '06': {'func': 'meshwork_size', 'params': "summary=True,verbose=False", 
                   'vis_stack': False},
            '07': {'func': 'save_estimated_params', 'params': "dest_dir=self.parameters['dest_dir']", 
                   'vis_stack': False},
            '08': {'func': 'nuke', 'params': None, 'vis_stack': False},
            '09': {'func': 'z_project_max', 'params': "substack=self.parameters['substack']",
                   'vis_stack': True, 'vis_params': "imtype='manipulated',save=True,dest_dir=self.parameters['dest_dir']"},
            '10': {'func': 'nuke', 'params': None,  'vis_stack': False}
            }


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
        
        self._all_filenames, self._all_filepaths = list_files_dir_str(self.root_path)

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
        Returns
        -------
        self.focal_planes : pandas DataFrame
            ??? 
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
            curr_planes = self.focal_planes[self.focal_planes['Type'].apply(lambda x: 'untrans' in x.lower())]
        elif 'car' in subdir.lower():
            curr_planes = self.focal_planes[self.focal_planes['Type'].apply(lambda x: 'car' in x.lower())]
        else:
            raise ValueError('Cell type not recognised as either Untransduced or CAR.')
        try: 
            any(curr_planes['File name'].sort_values() == sorted(curr_filenames))
        except ValueError: 
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



    def redefine_pipeline(self, new_pipeline):
        # which functions to call 
        # which intermediate results to save
        if not isinstance(new_pipeline, dict):
            raise TypeError('New pipeline must be specified as a dictionary.')
        self.pipeline_outline = new_pipeline


    def parametrise_pipeline(self, substack=None, theta=None, thetas=None, sigma=2, threshold=None, dest_dir=None):
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
        self.parameters = dict.fromkeys(['substack', 'theta', 'thetas', 'sigma', 'threshold', 'dest_dir'])
        for key, value in zip(self.parameters.keys(), [substack, theta, thetas, sigma, threshold, dest_dir]):
            self.parameters[key] = value


    def pipeline_construct(self, actin_img_instance, func):
        func_spcify = self.pipeline_outline[func]
        failed_funcs, failed_vis = [], []
    
        # try:
        #     eval(f"actin_img_instance.{func}({func_spcify['params']})")
        # except:
        #     failed_funcs.append(func)

        # if func_spcify['vis_stack']:
        #     try: 
        #         eval(f"actin_img_instance.visualise_stack({func_spcify['vis_params']})")    
        #     except:
        #         failed_vis.append(actin_img_instance.title)
        if func_spcify['params'] is not None: 
            eval(f"actin_img_instance.{func_spcify['func']}({func_spcify['params']})")
        else: 
            eval(f"actin_img_instance.{func_spcify['func']}()")
        if func_spcify['vis_stack']:
            eval(f"actin_img_instance.visualise_stack({func_spcify['vis_params']})")    



    def analysis_pipeline(self, filename, filepath):
        """ ??? """
        actin_img_instance = get_ActImg(filename, filepath)
        actin_img_instance.visualise_stack(imtype='original',save=True,dest_dir=self.__main_dest) 

        basal_stack, cyto_stack = self.focal_planes.loc[self.focal_planes['File name']==filename, ['Basal', 'Cytoplasmic']].values[0]
        ### use mean background intensity (extracted from image background in imagej)
        # basal_stack, cyto_stack, background = self.focal_planes.loc[self.focal_planes['File name']==filename, ['Basal', 'Cytoplasmic', 'Background']].values[0]
        # threshold = (float(background) - np.min(actin_img_instance.image_stack))/(np.max(actin_img_instance.image_stack)-np.min(actin_img_instance.image_stack))
        # self.parameters['threshold'] = threshold
        for stack, dest in zip([basal_stack, cyto_stack], [self.__basal_dest, self.__cyto_dest]):
            self.parameters['substack'] = stack
            self.parameters['dest_dir'] = dest
            for step in list(self.pipeline_outline.keys()):
                self.pipeline_construct(actin_img_instance, step)


    def visualise_html(self, subdir):
        """ ??? 
        Returns
        -------
        hidden: makes file named as ..... 
        """
        curr_filenames, curr_filepath = self._all_filenames[subdir], self._all_filepaths[subdir]
        curr_filenames = [f for f in curr_filenames if 'tif' in f and f not in self._filenames_to_del]

        all_output_types = dict.fromkeys(['original', 'max_proj', 'steer_gauss', 'min_proj', 'threshold'])

        out_types = ['original']
        type_file_ends = ['original']
        results_filenames = [res for res in os.listdir(self.__save_destdir+'/main') if 'png' in res]
        for key, val_check in zip(out_types, type_file_ends):
            all_output_types[key] = [res for res in results_filenames if val_check in res]


        out_types = ['max_proj', 'steer_gauss', 'min_proj', 'threshold']
        type_file_ends = ['max', '2order_thetas.png', 'min.png', 'threshold.png']
        results_filenames = [res for res in os.listdir(self.__save_destdir+'/basal') if 'png' in res]
        for key, val_check in zip(out_types, type_file_ends):
            all_output_types[key] = [res for res in results_filenames if val_check in res]

        md_filename = subdir[0:10]+'.md'
        results_base_dir = os.path.basename(self.__save_destdir)
        with open(os.path.join(self.__save_destdir,md_filename), 'w') as f:
            f.write('---\n')
            f.write(f'title: {subdir} basal and cytosolic meshwork results.')
            f.write('\n---\n\n\n')
            for i in range(len(curr_filenames)):
                f.write(f'# {curr_filenames[i]}')
                f.write('\n\n')
                if len(all_output_types['original']) > 0:
                    f.write(f'![](main/{all_output_types["original"][i]})'+'{ width=700px } ')
                f.write('\n\n')
                f.write('## Basal network  ')
                f.write('\n\n')
                f.write('**Maximum z-projection**  ')
                f.write('\n')
                if len(all_output_types['max_proj']) > 0:
                    f.write(f'![](basal/{all_output_types["max_proj"][i]})'+'{ height=300px }  ')
                f.write('\n')
                f.write('**Steerable second order Gaussian filter**  ')
                f.write('\n')
                if len(all_output_types['steer_gauss']) > 0:
                    f.write(f'![](basal/{all_output_types["steer_gauss"][i]})'+'{ height=300px }  ')
                f.write('\n')
                f.write('**Minimum z-projection**  ')
                f.write('\n')
                if len(all_output_types['min_proj']) > 0:
                    f.write(f'![](basal/{all_output_types["min_proj"][i]})'+'{ height=300px }  ')
                f.write('\n')
                f.write('**Binary thresholding**  ')
                f.write('\n')
                if len(all_output_types['threshold']) > 0:
                    f.write(f'![](basal/{all_output_types["threshold"][i]})'+'{ height=300px }  ')
                f.write('\n\n')
                f.write('## Cytosolic network  ')
                f.write('\n\n')
                f.write('**Maximum z-projection**  ')
                f.write('\n')
                if len(all_output_types['max_proj']) > 0:
                    f.write(f'![](cytosolic/{all_output_types["max_proj"][i]})'+'{ height=300px }  ')
                f.write('\n')
                f.write('**Steerable second order Gaussian filter**  ')
                f.write('\n')
                if len(all_output_types['steer_gauss']) > 0:
                    f.write(f'![](cytosolic/{all_output_types["steer_gauss"][i]})'+'{ height=300px }  ')
                f.write('\n')
                f.write('**Minimum z-projection**  ')
                f.write('\n')
                if len(all_output_types['min_proj']) > 0:
                    f.write(f'![](cytosolic/{all_output_types["min_proj"][i]})'+'{ height=300px }  ')
                f.write('\n')
                f.write('**Binary thresholding**  ')
                f.write('\n')
                if len(all_output_types['threshold']) > 0:
                    f.write(f'![](cytosolic/{all_output_types["threshold"][i]})'+'{ height=300px }  ')
                f.write('\n\n\n')
            

        pandoc_input = os.path.join(self.root_path, results_base_dir, md_filename)
        if shutil.which('pandoc') is None:
            warnings.warn(f'Pandoc is not installed. Returning only {pandoc_input.replace(".md", ".html")}.')
        else: 
            subprocess.run(f'pandoc -t slidy -s {pandoc_input} -o {pandoc_input.replace(".md", ".html")}', shell=True)


    def return_params(self):
        analysis_parameters = {'root_directory': self.root_path, 'parameters': self.parameters}
        with open(f'{os.path.join(self.root_path, os.path.basename(self.root_path))}.yml', 'w') as f:
            yaml.dump(analysis_parameters, f, default_flow_style=False)


    def run_analysis(self, visualise_as_html=False, return_parameters=False):
        """ Runs analysis on selected subdirs. """
        if self.parameters is None: 
            raise AttributeError('self.parameters has not been initialised; call self.parametrise_pipeline().')

        t_start = time.time()
        for subdir in tqdm(self.only_subdirs, desc='cell types'):

            curr_filenames, curr_filepath = self.initialise_curr_filenames_and_planes(subdir=subdir)
            self.initialise_res_dir(subdir=subdir)

            for name in tqdm(curr_filenames, desc='files'):
                self.analysis_pipeline(filename=name,filepath=curr_filepath)
        
            if visualise_as_html:
                self.visualise_html(subdir=subdir)

        delta_t = time.time() - t_start
        print(f'Analysis completed in {time.strftime("%H:%M:%S", time.gmtime(delta_t))}.')

        if return_parameters:
            self.return_params()


        
    def next(self):
        n=0
        raise NotImplementedError


