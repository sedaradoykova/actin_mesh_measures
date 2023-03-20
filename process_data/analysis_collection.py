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

###sample_data.run_analysis(visualise_as_html=True, return_parameters=False)


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




#### GETTING 305964 TABLES

# basal Untr

deconv_paths = [os.path.join(os.getcwd(), 'actin_meshwork_analysis\\process_data\\deconv_data\\', '_results_CARs'),
              os.path.join(os.getcwd(), 'actin_meshwork_analysis\\process_data\\deconv_data\\', '_results_Untr')]


names, paths = list_files_dir_str(deconv_paths[1])
filepaths = [os.path.join(paths['basal'], n) for n in names['basal'] if 'json' in n]
filenames = [n for n in names['basal'] if 'json' in n]
len(filepaths)
len(filenames)

# 1 min untransduced (basal)

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# get filenames in csv 
min1_fnames = sample_data.focal_planes['File name'].loc[(sample_data.focal_planes['File name'].str.contains('1min')) & 
                                                        sample_data.focal_planes['Type'].str.contains('Untrans')]
# get 1 min filepaths from basal filepaths 
min1_fpaths = [f for f in filepaths if '1min' in f]

len(min1_fpaths) == min1_fnames.shape[0]

(min1_fpaths[0].split("_params_")[-1].replace(".json", ""))

min1_Untr_basal_diams = []
min1_Untr_basal_dens = []
for filepath in min1_fpaths:
#    mtype = "_".join(filepath.split('\\')[-3:-1]).replace("_results_", '')
    with open(filepath, 'r') as file: 
        data = json.load(file)
    min1_Untr_basal_diams += data['equivalent_diameters']
    min1_Untr_basal_dens += [data['mesh_density_percentage']]

len(min1_Untr_basal_diams)

df_min1_Untr_basal_dens = pd.DataFrame({'file_name': min1_fnames,
                                         'mesh_density_percentage': min1_Untr_basal_dens})
df_min1_Untr_basal_diams = pd.DataFrame({'equiv_diameters': min1_Untr_basal_diams})


sns.histplot(x='equiv_diameters', data=df_min1_Untr_basal_diams)
plt.show()


sns.violinplot(y='mesh_density_percentage', data=df_min1_Untr_basal_dens, orient='h')
plt.show()

sns.violinplot(y='equiv_diameters', data=df_min1_Untr_basal_diams, orient='h')
plt.show()




### automate 

deconv_paths = [os.path.join(os.getcwd(), 'actin_meshwork_analysis\\process_data\\deconv_data\\', '_results_Untr'),
                os.path.join(os.getcwd(), 'actin_meshwork_analysis\\process_data\\deconv_data\\', '_results_CARs')]


min_Untr_basal_diams = []
min_Untr_basal_dens = []
df_equiv_diams = pd.DataFrame(columns=['file_name', 'time', 'cell_type','mesh_type','equiv_diameters'])
df_mesh_density = pd.DataFrame(columns=['file_name', 'time', 'cell_type','mesh_type','mesh_density'])
df_threshold_parameters = pd.DataFrame(columns=['file_name', 'time', 'cell_type','mesh_type','mean', 'std_dev'])
for (celltype, respath) in zip(['untrans', 'car'], deconv_paths):
    for meshtype in ['basal', 'cytosolic']:
        for time in ['1min', '3min', '8min']:

            names, paths = list_files_dir_str(respath)
            filepaths = [os.path.join(paths[meshtype], n) for n in names[meshtype] if 'json' in n]
            filenames = [n for n in names[meshtype] if 'json' in n]
            # get filenames in csv 
            min_fnames = sample_data.focal_planes['File name'].loc[(sample_data.focal_planes['File name'].str.contains(time)) & 
                                                                    sample_data.focal_planes['Type'].str.contains(celltype)]
            # get 1 min filepaths from basal filepaths 
            min_fpaths = [f for f in filepaths if time in f]

            for filepath in min_fpaths:
            #    mtype = "_".join(filepath.split('\\')[-3:-1]).replace("_results_", '')
                with open(filepath, 'r') as file: 
                    data = json.load(file)
                min_Untr_basal_diams += data['equivalent_diameters']
                min_Untr_basal_dens += [data['mesh_density_percentage']]

                name = (filepath.split("_params_")[-1].replace(".json", "")) 
                n = len(data['equivalent_diameters'])
                cell_type_pd = sample_data.focal_planes['Type'].loc[(sample_data.focal_planes['File name'].str.contains(name)) & 
                                                                    sample_data.focal_planes['Type'].str.contains(celltype, case=False)].tolist()

                mean, std_dev = data["aggregated_line_profiles"].values()

                df_equiv_diams = pd.concat([df_equiv_diams,
                                                pd.DataFrame({'file_name': [name]*n,
                                                'time': [time]*n,
                                                'cell_type': cell_type_pd*n,
                                                'mesh_type': [meshtype]*n,
                                                'equiv_diameters': data['equivalent_diameters']})],
                                                ignore_index=True, sort=False)
                df_mesh_density = pd.concat([df_mesh_density,
                                                pd.DataFrame({'file_name': [name],
                                                'time': [time],
                                                'cell_type': cell_type_pd,
                                                'mesh_type': [meshtype],
                                                'mesh_density': [data['mesh_density_percentage']]})],
                                                ignore_index=True, sort=False)
                df_threshold_parameters = pd.concat([df_threshold_parameters,
                                                pd.DataFrame({'file_name': [name],
                                                'time': [time],
                                                'cell_type': cell_type_pd,
                                                'mesh_type': [meshtype],
                                                'mean': [mean], 'std_dev': [std_dev]})],
                                                ignore_index=True, sort=False)
                

df_equiv_diams.shape[0] == len(min_Untr_basal_diams)


df_equiv_diams = df_equiv_diams.astype({'file_name': 'category','time': 'category', 'cell_type': 'category', 
                                                  'mesh_type': 'category', 'equiv_diameters': 'float64'})
df_mesh_density = df_mesh_density.astype({'file_name': 'category','time': 'category', 'cell_type': 'category', 
                                                  'mesh_type': 'category', 'mesh_density': 'float64'})
df_threshold_parameters = df_threshold_parameters.astype({'file_name': 'category','time': 'category', 'cell_type': 'category',
                                                          'mesh_type': 'category', 'mean': 'float64', 'std_dev': 'float64'})



#### PLOT EQUIV DIAMS 

plt.figure(figsize=(8,6))
sns.stripplot(x='cell_type', y='equiv_diameters', data=df_equiv_diams, hue='time', color="gray", edgecolor="black", alpha=0.1, dodge=True)
sns.violinplot(x='cell_type', y='equiv_diameters', data=df_equiv_diams, hue='time', dodge=True, alpha=0.75)
plt.show()


df_equiv_diams.cell_type.unique()

celltype_order = ['Untransduced','CAR_dual','CAR_antiCD19','CAR_antiCD22','CAR_UNTR']
time_order = ['1min','3min','8min']

g = sns.FacetGrid(df_equiv_diams, row='mesh_type', col='time',hue='mesh_type', col_order=time_order)
g.map(sns.violinplot, 'cell_type','equiv_diameters',dodge=True,alpha=0.75,order=celltype_order)
g.map(sns.stripplot, 'cell_type','equiv_diameters',color="gray",alpha=0.1,dodge=True,s=3, order=celltype_order)
#g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
axes = g.axes.flatten()
axes[0].set_title('Activation time: 1 min')
axes[1].set_title('Activation time: 3 min')
axes[2].set_title('Activation time: 8 min')
for n in np.arange(3,7):
    axes[n].set_title("")
axes[0].set_ylabel('Basal mesh size (equiv. diameter, px)')
axes[3].set_ylabel('Cytosolic mesh size (equiv. diameter, px)')
for ax in axes:
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.set_xlabel('Cell type')
plt.show()



g = sns.FacetGrid(df_equiv_diams, row='cell_type', col='time',hue='mesh_type', col_order=time_order)
g.map(sns.histplot,'equiv_diameters',alpha=0.75, legend=True)
#g.map(sns.stripplot, 'cell_type','equiv_diameters',color="lightgray",alpha=0.1,dodge=True,s=3, order=celltype_order)
plt.show()



### PLOT MESH DENSITY 

g = sns.FacetGrid(df_mesh_density, row='mesh_type', col='time',hue='mesh_type', col_order=time_order)
g.map(sns.violinplot, 'cell_type','mesh_density',dodge=True,alpha=0.75,order=celltype_order)
g.map(sns.stripplot, 'cell_type','mesh_density',color="lightgray",alpha=0.9,dodge=True,s=3, order=celltype_order)
#g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
axes = g.axes.flatten()
axes[0].set_title('Activation time: 1 min')
axes[1].set_title('Activation time: 3 min')
axes[2].set_title('Activation time: 8 min')
for n in np.arange(3,7):
    axes[n].set_title("")
axes[0].set_ylabel('Basal mesh density (%)')
axes[3].set_ylabel('Cytosolic mesh denisty (%)')
for ax in axes:
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.set_xlabel('Cell type')
plt.show()

g = sns.FacetGrid(df_mesh_density, row='cell_type', col='time',hue='mesh_type', col_order=time_order)
g.map(sns.histplot,'mesh_density',alpha=0.75, bins=10)
#g.map(sns.stripplot, 'cell_type','equiv_diameters',color="lightgray",alpha=0.1,dodge=True,s=3, order=celltype_order)
plt.show()

g = sns.FacetGrid(df_mesh_density, row='cell_type', col='time',hue='mesh_type', col_order=time_order)
g.map(sns.stripplot,'mesh_type', 'mesh_density',alpha=0.75)
#g.map(sns.stripplot, 'cell_type','equiv_diameters',color="lightgray",alpha=0.1,dodge=True,s=3, order=celltype_order)
plt.show()



### plot threshold valeus 

crosses = df_threshold_parameters.groupby(['cell_type','time'])[['mean','std_dev']].mean()
crosses = crosses.reset_index(level=[0,1])
crosses.dropna(axis = 0, how = 'any', inplace = True)

plt.figure(figsize=(8,6))
ax = sns.violinplot(x='cell_type', y='mean', data=df_threshold_parameters, hue='time', dodge=True, alpha=0.5)
sns.stripplot(x='cell_type', y='mean', data=df_threshold_parameters, hue='time', 
              color="black", edgecolor="black", alpha=0.7, dodge=True,ax=ax)
# sns.stripplot(x='cell_type', y='mean', data=crosses,marker='X',
#                 s=10, color='red', dodge=True)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:3], labels[0:3], loc=1, title='activation time') 
plt.title('threshold values: aggregted profile means')
plt.show()


plt.figure(figsize=(8,6))
ax = sns.violinplot(x='cell_type', y='std_dev', data=df_threshold_parameters, hue='time', dodge=True, alpha=0.5)
sns.stripplot(x='cell_type', y='std_dev', data=df_threshold_parameters, hue='time', 
              color="black", edgecolor="black", alpha=0.7, dodge=True,ax=ax)
# sns.stripplot(x='cell_type', y='mean', data=crosses,marker='X',
#                 s=10, color='red', dodge=True)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:3], labels[0:3], loc=1, title='activation time') 
plt.title('threshold values: aggregted profile std_devs')
plt.show()

### save csvs 

df_mesh_density.to_csv('all_mesh_densities_not_checked.csv')
df_equiv_diams.to_csv('all_equivalent_diameters_not_checked.csv')