""" This script filters all the data files in STED_stacks_Nov to find only ones containing 'beads' in them. 
It makes a new dir deconvolution_beads/beads_files and copies all matching files to that dir. 
"""

import os, shutil
from meshure.utils import search_files_root

rename_tiff_to_tif = False

# find files in 
data_path = os.path.join(os.getcwd(), "..", "STED_stacks_Nov")
assert os.path.exists(data_path)

# search dir for files matching 
beads = search_files_root(data_path, 'bead')

# check output files exist
if all([os.path.exists(f) for f in beads]): 
    print("All files exist.") 

if __name__ == '__main__':
    os.mkdir('deconvolution_beads/beads_files/')
    for file in beads: 
        shutil.copy2(file, 'deconvolution_beads/beads_files/')

    # rename files from tif to tiff
    file_names = os.listdir(os.path.join(os.getcwd(), 'deconvolution_beads', 'beads_files'))
    file_names = [f for f in file_names if f.endswith('.tiff')]

    if file_names and rename_tiff_to_tif: 
        for name in file_names: 
            os.rename(name, name.replace('tiff', 'tif'))
