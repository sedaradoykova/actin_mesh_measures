import os, cv2
from pathlib import Path
from PIL import Image
from PIL.TiffTags import TAGS
from itertools import chain
import numpy as np

""" TODO:
    - some todos down in docstrings 
    - read-in resolution and image dims from metadata
"""

def get_image_stack(file_path: str or Path, verbose: bool=False):
    """ Read in a user-defined tif file from a specified directory. 

    Returns
    -------
    image_stack : tuple 
        A tuple containing all slices of loaded image as np.ndarrays. 
    title : str
        The part of the file_path containing the file name. 
    """
    if not isinstance(file_path, (Path, str)):
        raise TypeError('file_path must me a Path or string instance.')
    title = os.path.split(file_path)[1]
    if '.tif' not in title:
        raise ValueError('File must be a tif/tiff file.')

    
    if os.path.exists(file_path):
        ret, image_stack = cv2.imreadmulti(file_path, [], cv2.IMREAD_UNCHANGED)
        if verbose:
            print("Total slices in stack: {n}".format(n=len(image_stack)))
        return image_stack, title
    else:
        raise FileNotFoundError('{f} not found.'.format(f=file_path))

def list_all_tiffs(root_dir: str):
    """ Returns list of all tif/tiff files in a directory.  
    """
    file_names = [x for x in os.listdir(root_dir) if 'tif' in x]
    print("Total tiffs in dir: {n}".format(n=len(file_names)))
    return file_names

def load_all_images(root_dir, file_names):
    """ Load all tif/tiffs from a directory. 

    See also
    --------
    search_files_root : Returns a list of files filtered by a keyword. 
    """
    # exec command --> make a dictionary and 
    # exec('{key}={value}'.format(key=key,value=value))
    # globals()['x{i}'.format(i=i)]=i
    # attributes of a class via setattributes(object, key, val).... 
    raise NotImplementedError

def get_meta(file_path):
    """ Returns a dictionary with tiff image metadata.
    """
    if not isinstance(file_path, (Path, str)): 
        raise TypeError('image_path must be a Path or string instance.')
    if '.tif' not in file_path:
        raise ValueError('Image must be a tif/tiff file.')

    img = Image.open(file_path)
    meta_dict = {TAGS[key] : img.tag[key] for key in img.tag_v2}
    return meta_dict

def get_resolution(meta: dict, nm: bool=False):
    """ Returns the resolution of an image given tiff file metadata. 

    Arguments
    --------- 
    meta : dict
        Metadata as read in from tiff files. Potentially works best with files which have been opened/processed with ImageJ.  
    nm : bool
        Defaults to False, returning resolution in microns. True returns resolution in nm. 
    """
    description = [val for val in meta['ImageDescription'][0].split('\n') if val] 
    description = {key: val for key, val in [item.split('=') for item in description]}

    res_x, unit_x = list(chain(*meta['XResolution']))
    res_y, unit_y = list(chain(*meta['YResolution']))
    if res_x==res_y and unit_x==unit_y:
        res_out = res_x/unit_x
        if nm:
            res_out = res_out*1e3
    else: 
        raise ValueError('Resolution is different in x and y.')
    return res_out

def get_axes_to_del(rows, cols, n):
    leftover = rows*cols-n
    if leftover:
        col, row = np.meshgrid(np.arange(cols),np.arange(rows)) 
        inds = [(r, c) for r,c in zip(row.ravel(), col.ravel())]
        return inds[-leftover:]
    else:
        return None
    

def get_fig_dims(n):
    if n == 2: 
        return (1,2, get_axes_to_del(1,2,n))
    elif n == 3:
        return (1,3, get_axes_to_del(1,3,n))
    if n <=4:
        return (2,2, get_axes_to_del(2,2,n))
    elif n <= 6:
        return (2,3, get_axes_to_del(2,3,n))
    elif n <= 9: 
        return (3,3, get_axes_to_del(3,3,n))
    else: 
        return (4, np.ceil(n/4), get_axes_to_del(4,np.ceil(n/4),n))



def list_files_dir_str(root_dir: str or Path): 
    """ Returns a dictionary with all subfolders (keys) and their files listed (values).
    Note: files in the root directory are not returned. 
    todo input validation

    Arguments
    ---------
    root_dir : str or Path
        A directory which contains subdirectories and files for analysis. 

    Returns
    -------
    dict_files : dict
        A dictionary mapping a key to filenames contained in that dir.
    dict_paths : dict
        A dictionary mapping a key to paths of the parent folders of the files in dict_files - their keys match.
    """
    paths, bases, files = [], [], []

    for root, dirs, files_in in os.walk(root_dir):
        path = root.split(os.sep) 
        paths.append(path) # full paths 
        bases.append(os.path.basename(root)) # final dir 
        files.append([file for file in files_in]) # files

    keys = []
    for p in paths:
        len_0 = len(": ".join(paths[0]))+2
        p_out = ": ".join(p)
        length = len(p_out)
        keys.append(p_out[len_0:length+1:])

    dict_files = {key: val for key, val in zip(keys, files) if val and key} # only if values and key are present
    dict_paths = {key: os.path.join(*val).replace('C:', "C:\\") for key,val in zip(keys, paths) if key in dict_files.keys()}
    return dict_files, dict_paths

def filter_files(dict_files: str or Path, dict_paths: str or Path, keyword: str):
    """ Given a dictionary of filenames and another dictionary with (matching keys) of parent dirs, 
    returns the full paths of files containing keyword.
    Note: case insensitive (will return all results irrespective of upper/lower case). 
    
    TODO: problem with output strings C:users 
        check outputs for list_files_dir_str and how the strings are joined

    ...
    Arguments
    ---------
    dict_files : dict
        A dictionary mapping a key to filenames contained in that dir e.g. the output of list_files_dir_str().
    dict_paths : dict
        A dictionary mapping a key to paths of the parent folders of the files in dict_files - their keys match.
    
    Returns
    -------
    list

    See also
    --------
    list_files_dir_str : A function which yields suitable input.

    """ 

    #[[key, file] for key, val in dict_files.items() for file in val if "bead" in file.lower()]
    res = {}
    for key, val in  dict_files.items():
        for file in val:
            if keyword.lower() in file.lower():
                if key not in res.keys():
                    #res[key] = [file]
                    res[key] = [os.path.join(dict_paths[key], file)]
                else:
                    #res[key].append(file)
                    res[key].append(os.path.join(dict_paths[key], file))
    return res


def search_files_root(root_dir: str, keyword: str):
    """ Wrapper which processes the output of filter_files() to a list of paths. """
    d_files, d_paths = list_files_dir_str(root_dir)
    matching_file_dict = filter_files(d_files, d_paths, keyword)
    matching_paths = list(chain(*matching_file_dict.values()))
    if all([os.path.exists(f) for f in matching_paths]): 
        return matching_paths
    else: 
        raise FileNotFoundError('Some files in output were not found. Please check input.')


