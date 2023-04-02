import os, csv, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from meshure.actimg_collection import ActImgCollection, list_files_dir_str

resdir = 'actin_meshwork_analysis/process_data/deconv_data/_results_CARs'
focal_dir = 'actin_meshwork_analysis/scratch_misc/params/types_to_elongate.csv'
destdir = 'actin_meshwork_analysis/scratch_misc/params'

focal_planes = pd.read_csv(focal_dir)
focal_planes.head()

focal_planes.value_counts('Type')

filenames, _ = list_files_dir_str(resdir)


ind = {'Maxiumum projection': 'max', 
             'Steerable filter response': '2order_thetas.png', 
             'Minimum projection': 'min.png',
             'Thresholded image': 'threshold.png', 
             'Segmented mesh': '_mesh_segmentation'}

ind = ind['Maxiumum projection', 'Minimum projection', 'Segmented mesh']

with open(os.path.join(resdir, 'test.md'), 'w') as f:
    f.write('---\n')
    f.write(f'title: {os.path.basename(resdir)} basal and cytosolic meshwork results.')
    f.write('\n---\n\n\n')
    for file in focal_planes[focal_planes.Type!='Untransduced']['File name']:
        fname = file.split('.')[0]
        fstack = [f for f in filenames['main'] if fname in f][0]
        f.write(f'# {fname}')
        f.write('\n\n')
        f.write(f'![](main/{fstack})'+'{ width=700px }  ')
        f.write('\n\n')
        for cdir in ('basal', 'cytosolic'):
            all_cnames = [f for f in filenames[cdir] if fname in f]
            f.write(f'## {cdir} network  ')
            f.write('\n\n')
            for subtitle, ending in ind.items():
                try: 
                    curr_cname = [f for f in all_cnames if ending in f]
                    if len(curr_cname) == 1: 
                        f.write(f'**{subtitle}**  ')
                        f.write('\n')
                        f.write(f'![]({cdir}/{curr_cname[0]})'+'{ height=300px }  ')
                        f.write('\n\n')
                except NameError:
                    pass

