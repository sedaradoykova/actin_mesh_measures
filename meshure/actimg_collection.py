import os, csv, yaml, time, subprocess, shutil, warnings
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
from meshure.actimg import get_ActImg
from meshure.actimg_binary import get_ActImgBinary
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
    pipeline_outline = {# loop through these (if needed) for basal/cytoplasmic results 
            '01': {'func': 'z_project_max', 'params': "substack=self.parameters['substack']",
                   'vis_stack': True, 'vis_params': "imtype='manipulated',save=True,dest_dir=self.parameters['dest_dir']"},
            '02': {'func': 'nuke', 'params': None, 'vis_stack': False},
            '03': {'func': 'normalise', 'params': None, 
                   'vis_stack': False, 'vis_params': "imtype='manipulated',save=True,dest_dir=self.parameters['dest_dir']"},
            '04': {'func': 'steerable_gauss_2order_thetas', 
                   'params': "thetas=self.parameters['thetas'],sigma=self.parameters['sigma'],substack=self.parameters['substack'],visualise=False",
                   'vis_stack': False, 'vis_params': "imtype='manipulated',save=True,dest_dir=self.parameters['dest_dir']"},
            '05': {'func': 'z_project_min', 'params': None,
                   'vis_stack': True, 'vis_params': "imtype='manipulated',save=True,dest_dir=self.parameters['dest_dir']"},
            '06': {'func': 'threshold_dynamic', 'params': "std_dev_factor=0,return_mean_std_dev=False",
                   'vis_stack': False, 'vis_params': "imtype='manipulated',save=True,dest_dir=self.parameters['dest_dir']"},
            # '07': {'func': 'meshwork_density', 'params': "verbose=False", 
            #        'vis_stack': False},
            # '08': {'func': 'meshwork_size', 'params': "summary=True,verbose=False,save_vis=True,dest_dir=self.parameters['dest_dir']", 
            #        'vis_stack': False},
            # '09': {'func': 'surface_area', 'params': "verbose=False", 
            #        'vis_stack': False},
            # '10': {'func': 'save_estimated_params', 'params': "dest_dir=self.parameters['dest_dir']", 
            #        'vis_stack': False},
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
        self.failed_segmentations = []



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
            File name = name of tiff file 
            Type = cell type, either 'Untransduced' or 'CAR'
            Basal = one or two comma separated integers specifying the focal plane (or focal plane [start, end])
                These are converted to a list inside the function
            Cytoplasmic = one or two comma separated integers specifying the focal plane (or focal plane [start, end])
                These are converted to a list inside the function
            Notes = field not used by this function. 
        
        Returns
        -------
        self.focal_planes : pandas DataFrame
            ??? 
        
        Raises
        -----
        Warning
            If the filenames in the CSV file are not unique, a warning is raised but the program is executed. 
        """
        if not isinstance(filepath, str):
            raise TypeError('filepath path must be a string.')
        if not os.path.exists(filepath):
            raise ValueError(f'Path not found: {filepath}.')
        if not filepath.endswith('.csv'): 
            raise TypeError(f'Invalid file {os.path.basename(filepath)}; CSV required.')

        focal_planes = pd.read_csv(filepath)
        # if list(focal_planes.columns) != ['File name', 'Basal', 'Cytoplasmic', 'Notes']:
        #     raise ValueError("Csv file contains unexpected columns. Columns must be named ['File name', 'Type', 'Basal', 'Cytoplasmic', 'Notes']")

        # if basal or cytoplasmic planes are not speficied, discard   
        self._filenames_to_del = focal_planes['File name'][focal_planes[['Basal', 'Cytoplasmic']].isna().any(axis=1)].tolist()
        focal_planes = focal_planes.dropna(subset=['Basal', 'Cytoplasmic'])

        focal_planes['Basal'] = focal_planes['Basal'].apply(lambda val: [int(x) for x in val.split(',')])
        focal_planes['Cytoplasmic'] = focal_planes['Cytoplasmic'].apply(lambda val: [int(x) for x in val.split(',')])


        if focal_planes.iloc[:,0].unique().shape[0] == focal_planes.shape[0]:
            warnings.warn(f'File names in {os.path.basename(filepath)} are not unique.')

        self.focal_planes=focal_planes
        return None


    def _initialise_curr_filenames_and_planes(self, subdir):
        """ Called inside analysis loop. Subsets the current filenames which are being processed based on subdir. 
        
        Arguments
        ---------
        subdir : str 
            ?????? 
        
        Returns
        -------
        curr_filenames : list
        curr_filepath : list

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
            return curr_filenames, curr_filepath, curr_planes


    def _initialise_res_dir(self, subdir):
        """ Initialise results directory. """
        self.__save_destdir = self.res_path if self.res_path else os.path.join(self.root_path, '_results_'+subdir)
        if not os.path.exists(self.__save_destdir):
            os.mkdir(self.__save_destdir)

        self.__basal_dest = os.path.join(self.__save_destdir, 'basal')
        if not os.path.exists(self.__basal_dest):
            os.mkdir(self.__basal_dest)

        self.__cyto_dest = os.path.join(self.__save_destdir, 'cytoplasmic')
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


    def _pipeline_construct(self, actin_img_instance, func):
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

    def _extract_parameters(self, actimgbinary):
        if 'activation_time' not in actimgbinary.estimated_parameters['cell_type'].keys():
            actimgbinary.estimated_parameters['cell_type']['activation_time'] = actimgbinary._get_activation_time()
        nrep = actimgbinary.estimated_parameters['mesh_holes']['hole_parameters'].shape[0]
        unit = actimgbinary.estimated_parameters['mesh_holes']['unit']
        df = pd.DataFrame({
            'filename': np.repeat(actimgbinary.title, nrep),
            'cell_type': np.repeat(actimgbinary.estimated_parameters['cell_type']['type'], nrep),
            'mesh_type': np.repeat(actimgbinary.estimated_parameters['cell_type']['mesh_type'], nrep),
            'activation_time': np.repeat(actimgbinary.estimated_parameters['cell_type']['activation_time'], nrep),
            f'cell_surface_area_{actimgbinary.estimated_parameters["cell_surface_area"]["unit"]}': 
            np.repeat(actimgbinary.estimated_parameters['cell_surface_area']['area'], nrep),
            'mesh_density': actimgbinary.estimated_parameters['mesh_density'],
            'periph_mesh_density': actimgbinary.estimated_parameters['peripheral_mesh_density'],
            f'equivalent_diameter_area_{unit}': actimgbinary.estimated_parameters['mesh_holes']['hole_parameters'][f'equivalent_diameter_area_{unit}'],
            f'area_{unit}': actimgbinary.estimated_parameters['mesh_holes']['hole_parameters'][f'area_{unit}'],
            f'perimeter_{unit.split("^")[0]}': actimgbinary.estimated_parameters['mesh_holes']['hole_parameters'][f'perimeter_{unit.split("^")[0]}']}) 
        self.all_dfs_list.append(df) 
        

    def _analysis_pipeline(self, filename, filepath, curr_planes):
        """ ??? """
        actin_img_instance = get_ActImg(filename, filepath)
        actin_img_instance.visualise_stack(imtype='original',save=True,dest_dir=self.__main_dest) 

        basal_stack, cyto_stack, cell_type = curr_planes.loc[curr_planes['File name']==filename, ['Basal', 'Cytoplasmic', 'Type']].values[0]
        for mesh_type, stack, dest in zip(['Basal', 'Cytoplasmic'], [basal_stack, cyto_stack], [self.__basal_dest, self.__cyto_dest]):
            self.parameters['substack'] = stack
            self.parameters['dest_dir'] = dest
            for step in list(self.pipeline_outline.keys()):
                self._pipeline_construct(actin_img_instance, step)
            actimgbinary = get_ActImgBinary(actin_img_instance)
            actimgbinary.estimated_parameters['cell_type'] = {'type': cell_type, 'mesh_type': mesh_type}
            try:
                actimgbinary.surface_area(n_dilations_erosions=(0,2),closing_structure=None,extra_dilate_fill=True)
                actimgbinary.mesh_holes_area()
                actimgbinary.visualise_segmentation(save=True, dest_dir=dest)
                actimgbinary.mesh_density()
                actimgbinary.quantify_mesh()
                actimgbinary.peripheral_mesh_density()
                #actimgbinary.save_estimated_parameters(dest)
                actimgbinary.save_log(dest_dir=self.__save_destdir, dest_file='failed_segmentation_logs.txt')
            except:
                self.failed_segmentations.append(f'{actimgbinary.title}--{mesh_type}')
            else: 
                self._extract_parameters(actimgbinary)
            finally:
                actin_img_instance.nuke()


    def _visualise_html(self, subdir: str, curr_planes: pd.DataFrame, include_steps: list=None):
        """ ??? 
        Returns
        -------
        hidden: makes file named as ..... 
        """
        curr_filenames, curr_filepath = self._all_filenames[subdir], self._all_filepaths[subdir]
        curr_filenames = [f for f in curr_filenames if 'tif' in f and f not in self._filenames_to_del]

        filenames, _ = list_files_dir_str(self.__save_destdir)

        md_filename = subdir[0:10]+'.md' if len(subdir) >= 10 else subdir+'.md'
        results_base_dir = os.path.basename(self.__save_destdir)

        all_outpts = {'Maximum projection': 'max', 
                    'Steerable filter response': '2order_thetas.png', 
                    'Minimum projection': 'min.png',
                    'Thresholded image': 'threshold.png', 
                    'Segmented mesh': '_mesh_segmentation'}

        subdict = {key: val for key, val in all_outpts.items() if key in include_steps} if include_steps else all_outpts

        with open(os.path.join(self.__save_destdir, md_filename), 'w') as f:
            f.write('---\n')
            f.write(f'title: {results_base_dir} basal and cytoplasmic meshwork results.')
            f.write('\n---\n\n\n')
            for file in curr_planes['File name']:
                fname = file.split('.')[0]
                fstack = [f for f in filenames['main'] if fname in f][0]
                f.write(f'# {fname}')
                f.write('\n\n')
                f.write(f'![](main/{fstack})'+'{ width=700px }  ')
                f.write('\n\n')
                for cdir in ('basal', 'cytoplasmic'):
                    all_cnames = [f for f in filenames[cdir] if fname in f]
                    f.write(f'## {cdir.title()} network  ')
                    f.write('\n\n')
                    for subtitle, ending in subdict.items():
                        try: 
                            curr_cname = [f for f in all_cnames if ending in f]
                            if len(curr_cname) == 1: 
                                f.write(f'**{subtitle}**  ')
                                f.write('\n')
                                f.write(f'![]({cdir}/{curr_cname[0]})'+'{ height=300px }  ')
                                f.write('\n\n')
                        except NameError:
                            pass


                f.write('\n\n\n')
            

        pandoc_input = os.path.join(self.root_path, results_base_dir, md_filename)
        if shutil.which('pandoc') is None:
            warnings.warn(f'Pandoc is not installed. Returning only {pandoc_input}.')
        else: 
            subprocess.run(f'pandoc -t slidy -s {pandoc_input} -o {pandoc_input.replace(".md", ".html")}', shell=True)


    def _return_params(self):
        analysis_parameters = {'root_directory': self.root_path, 'parameters': self.parameters}
        with open(f'{os.path.join(self.root_path, os.path.basename(self.root_path))}.yml', 'w') as f:
            yaml.dump(analysis_parameters, f, default_flow_style=False)


    def run_analysis(self, visualise_as_html: bool=False, return_parameters: bool=False, save_as_single_csv: bool=True):
        """ Runs analysis on selected subdirs. """
        if self.parameters is None: 
            raise AttributeError('self.parameters has not been initialised; call self.parametrise_pipeline().')

        t_start = time.time()
        self.all_dfs_list = []
        for subdir in tqdm(self.only_subdirs, desc='cell types'):

            curr_filenames, curr_filepath, curr_planes = self._initialise_curr_filenames_and_planes(subdir=subdir)
            self._initialise_res_dir(subdir=subdir)

            for name in tqdm(curr_filenames, desc='files'):
                self._analysis_pipeline(filename=name,filepath=curr_filepath,curr_planes=curr_planes)
        
            if visualise_as_html:
                self._visualise_html(subdir=subdir, curr_planes=curr_planes, include_steps=['Maximum projection','Minimum projection','Segmented mesh'])

            if save_as_single_csv: 
                self.all_parameters_df = pd.concat(self.all_dfs_list)
                self.all_parameters_df.to_csv(
                    os.path.join(self.__save_destdir, '../all_params_csv.csv'),sep=',',index=False,header=True)
                    
        delta_t = time.time() - t_start
        print(f'Analysis completed in {time.strftime("%H:%M:%S", time.gmtime(delta_t))}.')



        n_failed = len(self.failed_segmentations)
        if n_failed > 0:
            response = input(f'Some segmentations failed, see saved logs. View {n_failed} failed cases? [y/n]\n')
            if response == ('y','Y'):
                [print(name) for name in self.failed_segmentations]
            elif response == ('n','N'):
                pass
            else: 
                'Invalid input, step passed.' 

        if return_parameters:
            self._return_params()


        
    def next(self):
        n=0
        raise NotImplementedError


