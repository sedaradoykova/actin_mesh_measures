""" This script filters all the data files in a working directory to clear unwanted files. """

import os, shutil, itertools
from actin_meshwork_analysis.meshwork.utils import search_files_root, list_files_dir_str


# find files in 
data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/deconv_data")
os.path.exists(data_path)

# search dir for files matching 
matches = search_files_root(data_path, 'B7')

# check output files exist
if matches and all([os.path.exists(f) for f in matches]): 
    print("All files exist.") 


if __name__ == '__main__':
    if matches: 
        for file in matches: 
            os.remove(file)


all_filenames, all_filepaths = list_files_dir_str(data_path)

def print_summary(all_filenames):
    for cat in ['1min', '3min', '8min']:
        for key, vals in all_filenames.items():
            print(key, " "*(20-len(key)), ":", cat, " :", len([val for val in vals if cat in val]))
    print("\n")
    for key, vals in all_filenames.items():
        print(key, " "*(20-len(key)), ": total :", len(vals))

    print("\nTotal files", " "*17, ":", len(search_files_root(data_path, '.tif')))

print_summary(all_filenames)


# This data needs to be further processed, as there are multiple cells in some FOVs.
# 
# CARs_8.11.22          : 1min  : 9
# Untransduced_1.11.22  : 1min  : 10
# CARs_8.11.22          : 3min  : 9
# Untransduced_1.11.22  : 3min  : 6
# CARs_8.11.22          : 8min  : 7
# Untransduced_1.11.22  : 8min  : 5

# CARs_8.11.22          : total : 26
# Untransduced_1.11.22  : total : 21

# Total files                   : 46