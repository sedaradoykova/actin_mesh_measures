import os
import numpy as np
from meshure.actimg_collection import ActImgCollection, list_files_dir_str


# data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/sample_data")
# focal_plane_filename = os.path.join(data_path, 'sample_focal_planes.csv')
# only_subdirs = ['Untransduced','CARs']
## 15/03 Analysis completed in 00:00:57.

data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/deconv_data")
focal_plane_filename = os.path.join(data_path, 'basal_cytosolic_focal_planes_CARs.csv')
only_subdirs = ['Untransduced_1.11.22_processed_imageJ','CARs_8.11.22_processed_imageJ']
## OLD Analysis completed in 00:07:47.
# 15/03 Analysis completed in 00:10:28.

#theta_x6 = np.arange(0,360,60)
#theta_x3 = np.arange(60,360,120)
theta_x20 = np.linspace(0,171,20)
parameters = [None, None, theta_x20, 2, None, None]

sample_data = ActImgCollection(root_path=data_path)
sample_data.print_summary(only_subdirs)
sample_data.get_focal_planes(focal_plane_filename)
sample_data.only_subdirs = only_subdirs
sample_data.parametrise_pipeline(*parameters)
#sample_data.parametrise_pipeline(None, None, theta_x6, 2, None) ### for background threshold

sample_data.run_analysis(visualise_as_html=True, return_parameters=False)


#### some interactive features to be included further 
# sample_data = ActImgCollection(root_path=data_path)
# sample_data.print_summary(interact=True)


# for labelled cars 
sample_data.focal_planes.Type.value_counts()

import pandas as pd
tab = pd.DataFrame({
    'Type': list(sample_data.focal_planes['Type'][sample_data.focal_planes['Type'].apply(lambda x: 'car' in x.lower())]),
    'Time': [item.split('_')[0] for item in list(sample_data.focal_planes['File name'][sample_data.focal_planes['Type'].apply(lambda x: 'car' in x.lower())])]
            })
tab.value_counts()

"""
Untransduced    22
CAR_dual        25
CAR_antiCD19    11
CAR_antiCD22     7
CAR_UNTR         6

Untransduced               : 1min  : 11
Untransduced               : 3min  : 6
Untransduced               : 8min  : 5
Untransduced               : total : 22

CARs                       : 1min  : 16
CARs                       : 3min  : 19
CARs                       : 8min  : 17
CARs                       : total : 53

Type            Time    Count

Untransduced    1min     11
                3min     6
                8min     5
                 Total   22

CAR_dual        1min     11
                3min     8
                8min     6
                 Total   25

CAR_antiCD19    1min     None
                3min     7
                8min     4
                 Total   11

CAR_antiCD22    1min     2
                3min     1
                8min     4
                 Total   7

CAR_UNTR        1min     None
                3min     3
                8min     3
                 Total   6

"""

# for labelled cars, postprocessing will be required  
sample_data.focal_planes.Type.unique()


# pandoc -t slidy _results_Untr/Untransduc.md _results_CARs/CARs.md  -o results.html
## doesn't work because of duplicate identifiers.... 


## post processing of parameters 
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# sample_paths = [os.path.join(os.getcwd(), 'actin_meshwork_analysis\\process_data\\sample_data\\', '_results_CARs'), 
#                 os.path.join(os.getcwd(), 'actin_meshwork_analysis\\process_data\\sample_data\\', '_results_Untr')]
deconv_paths = [os.path.join(os.getcwd(), 'actin_meshwork_analysis\\process_data\\deconv_data\\', '_results_CARs'),
              os.path.join(os.getcwd(), 'actin_meshwork_analysis\\process_data\\deconv_data\\', '_results_Untr')]

# get filenames and filepaths 
filenames = []
filepaths = []
for rpath in deconv_paths:
    names, paths = list_files_dir_str(rpath)
    for lab in ['basal', 'cytosolic']:
        #[paths[lab]]
        filepaths += [os.path.join(paths[lab], n) for n in names[lab] if 'json' in n]
        filenames += [n for n in names[lab] if 'json' in n]



# open json and extract mean and median mesh sizes
respd = pd.DataFrame(columns=['mesh_type', 'mean', 'median'])
for filepath, filename in zip(filepaths, filenames):
    mtype = "_".join(filepath.split('\\')[-3:-1]).replace("_results_", '')
    with open(filepath, 'r') as f: 
        data = json.load(f)
    respd.loc[filepath, 'mesh_type'] = mtype
    respd.loc[filepath,'mean'] = data['mesh_size_summary']['mean']
    respd.loc[filepath,'median'] = data['mesh_size_summary']['median']


#respd.set_index()

# rotate horizontally 
respd_pv = respd.pivot(columns='mesh_type')


# plotting 
respd.plot(x='mesh_type', y='mean',kind='scatter', subplots=True)
plt.show()


# categorial scatter plot by mesh and cell type
sns.catplot(x='mesh_type', y='mean', data=respd, hue='mesh_type')
plt.show()


respd_long = respd.melt(id_vars=['mesh_type'])

g = sns.FacetGrid(respd_long, col='variable', hue='mesh_type')
g.map(sns.scatterplot, 'mesh_type', 'value', alpha=.7)
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh Type')
axes = g.axes.flatten()
axes[0].set_title('Mean mesh size')
axes[1].set_title('Median mesh size')
axes[0].set_ylabel('Mesh Size (equiv. diameter, px)')
for ax in axes:
    ax.set_xlabel('Mesh Type')
plt.show()



respd_long = respd_long.astype({'mesh_type': 'category', 'variable': 'category', 'value': 'float64'})




import scipy.stats as stats
# stats f_oneway functions takes the groups as input and returns ANOVA F and p value
cars_basal = respd_long.value[(respd_long.variable=='median') & (respd_long.mesh_type=='CARs_basal')]
cars_cytosolic = respd_long.value[(respd_long.variable=='median') & (respd_long.mesh_type=='CARs_cytosolic')]
untr_basal = respd_long.value[(respd_long.variable=='median') & (respd_long.mesh_type=='Untr_basal')]
untr_cytosolic = respd_long.value[(respd_long.variable=='median') & (respd_long.mesh_type=='Untr_cytosolic')]

fvalue, pvalue = stats.f_oneway(cars_basal, cars_cytosolic, untr_basal, untr_cytosolic)
print(f'F value:  {fvalue:.4f},    p value:  {pvalue:.6f}')


g = sns.violinplot(x='variable', y='value', data=respd_long, hue='mesh_type', alpha=0.7)
g.set_xlabel('Measurement type')
g.set_ylabel('Mesh size (equiv. diameter, px)')
g.set_title(f'one-way ANOVA of medians (F_3 = {fvalue:.4f}, p = {pvalue:.6f})')
plt.legend(title='Mesh Type', loc='upper right')
plt.show()



import statsmodels.api as sm
from statsmodels.formula.api import ols
# Ordinary Least Squares (OLS) model
model = ols('value ~ C(mesh_type)', data=respd_long[respd_long.variable=='median']).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table

