import pytest, os
from meshwork.utils import list_files_dir_str

""" TODO: implement a dummy dir structure to test this on. """

def test_list_files_dir_str_output():
    result = list_files_dir_str(r'C:\Users\sedar\Documents\UCL\1_BIOL0041_MSci_project\STED_stacks_Nov')
    assert result['UNT: Fri'][:5:] == [
        'UNT_Fri.lif - 1min_B7-1_Fri_FOV1.tif', 
        'UNT_Fri.lif - 1min_B7-1_Fri_FOV2.tif', 
        'UNT_Fri.lif - 1min_Fri_FOV1.tif', 
        'UNT_Fri.lif - 1min_Fri_FOV2.tif', 
        'UNT_Fri.lif - 1min_Fri_FOV3.tif']

def test_list_files_dir_keys():
    d_files, d_dirs = list_files_dir_str('C:\\Users\\sedar\\Documents\\UCL\\1_BIOL0041_MSci_project\\STED_stacks_Nov')
    assert d_files.keys() == d_dirs.keys()
