import os
import numpy as np
from meshure.actimg_collection import ActImgCollection, list_files_dir_str


# data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/sample_data")
# focal_plane_filename = os.path.join(data_path, 'sample_focal_planes.csv')
# only_subdirs = ['Untransduced','CARs']
## 15/03 Analysis completed in 00:00:57.
## 02/04 Analysis completed in 00:01:23.
## 03/04 Analysis completed in 00:01:56.
## 04/04 Analysis completed in 00:01:20.

data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/deconv_data")
focal_plane_filename = os.path.join(data_path, 'basal_cytosolic_focal_planes_CARs_noCAR_untr.csv')
only_subdirs = ['Untransduced_1.11.22_processed_imageJ','CARs_8.11.22_processed_imageJ']
## OLD Analysis completed in 00:07:47.
## 15/03 Analysis completed in 00:10:28.
## 11/03 Analysis completed in 00:17:50.
## 04/04 Analysis completed in 00:12:14.

#theta_x6 = np.arange(0,360,60)
#theta_x3 = np.arange(60,360,120)
theta_x20 = np.linspace(0,180,21)
parameters = [None, None, theta_x20, 2, None, None]

sample_data = ActImgCollection(root_path=data_path)
sample_data.print_summary(only_subdirs)
sample_data.get_focal_planes(focal_plane_filename)
sample_data.only_subdirs = only_subdirs
sample_data.parametrise_pipeline(*parameters)
#sample_data.parametrise_pipeline(None, None, theta_x6, 2, None) ### for background threshold

sample_data.run_analysis(visualise_as_html=True, return_parameters=False, save_as_single_csv=True)


#### some interactive features to be included further 
# sample_data = ActImgCollection(root_path=data_path)
# sample_data.print_summary(interact=True)


# for labelled cars - count 
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

WITHOUT THE DODGY AND UNTR 
CAR     1min    13      Untransduced    1min     11
        3min    14                      3min     6
        8min    14                      8min     5
        Total:  41                      Total:   22

"""

# for labelled cars, postprocessing will be required  
sample_data.focal_planes.Type.unique()


# pandoc -t slidy _results_Untr/Untransduc.md _results_CARs/CARs.md  -o results.html
## doesn't work because of duplicate identifiers.... 


## post processing of parameters 


#### GETTING 305964 TABLES

# respd_pv = respd.pivot(columns='mesh_type')
# respd_long = respd.melt(id_vars=['mesh_type'])


### automate
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
                min_Untr_basal_diams += data['estimated_parameters']['equivalent_diameters']
                min_Untr_basal_dens += [data['estimated_parameters']['mesh_density_percentage']]

                name = (filepath.split("_params_")[-1].replace(".json", "")) 
                n = len(data['estimated_parameters']['equivalent_diameters'])
                cell_type_pd = sample_data.focal_planes['Type'].loc[(sample_data.focal_planes['File name'].str.contains(name)) & 
                                                                    sample_data.focal_planes['Type'].str.contains(celltype, case=False)].tolist()

                mean, std_dev = data['estimated_parameters']["aggregated_line_profiles"].values()
                surface_area = data['estimated_parameters']

                df_equiv_diams = pd.concat([df_equiv_diams,
                                                pd.DataFrame({'file_name': [name]*n,
                                                'time': [time]*n,
                                                'cell_type': cell_type_pd*n,
                                                'mesh_type': [meshtype]*n,
                                                'equiv_diameters': data['estimated_parameters']['equivalent_diameters']})],
                                                ignore_index=True, sort=False)
                df_mesh_density = pd.concat([df_mesh_density,
                                                pd.DataFrame({'file_name': [name],
                                                'time': [time],
                                                'cell_type': cell_type_pd,
                                                'mesh_type': [meshtype],
                                                'mesh_density': [data['estimated_parameters']['mesh_density_percentage']]})],
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

### CONVERT CARs to CAR 

for x in [df_equiv_diams, df_mesh_density, df_threshold_parameters]:
    x['cell_type'] = ['Untransduced' if 'untransd' in entry.lower() else 'CAR' for entry in x['cell_type']]



# segmentation success

sample_data.focal_planes[['Basal_mesh', 'Cytosolic_mesh']].value_counts()
sample_data.focal_planes['Basal_mesh'].value_counts()
sample_data.focal_planes['Cytosolic_mesh'].value_counts()

sample_data.focal_planes[['Basal_mesh', 'Cytosolic_mesh', 'Type']].value_counts()

"""TOTAL = 63
Basal_mesh  Cytosolic_mesh
    v           v                 45
    x           v                  9
    v           x                  4
    x           x                  5

    Basal_mesh      Cytosolic_mesh
v           49      54
x           14      9

Basal_mesh  Cytosolic_mesh  Type        
v           v               CAR             30      = 18+7+5 (CAR_dual+CAR_antiCD19+CAR_antiCD22)
                            Untransduced    15

x           v               CAR             5       = 4+1 (CAR_dual+CAR_antiCD22)
                            Untransduced    4

v           x               CAR              3      = 2+1 (CAR_antiCD19_CAR_dual)     
                            Untransduced     1

x           x               CAR              3      = 1+1+1 (CAR_antiCD19+CAR_antiCD22+CAR_dual)
                            Untransduced     2
"""

sample_data.focal_planes.loc[(sample_data.focal_planes['Basal_mesh']=='x') | 
                             (sample_data.focal_planes['Cytosolic_mesh']=='x')][['File name', 'Type', 'Basal', 'Cytosolic']]






#### PLOT EQUIV DIAMETERS

# scatter 
g = sns.FacetGrid(df_equiv_diams, col='time')
g.map_dataframe(sns.stripplot,'cell_type', 'equiv_diameters', hue='mesh_type', 
                alpha=0.75, dodge=True, palette='muted')
g.set_xlabels('')
g.set_ylabels('Mesh density size (equivalent diameter, nm)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()

# box + scatter
g = sns.FacetGrid(df_equiv_diams, col='time', col_order=['1min','3min','8min'])
g.map_dataframe(sns.barplot, 'cell_type', 'equiv_diameters', hue='mesh_type', dodge=True, palette='muted',
                estimator=np.median, errorbar=('pi',50), capsize=0.2)
g.map_dataframe(sns.stripplot,'cell_type', 'equiv_diameters', hue='mesh_type', 
                alpha=0.5, dodge=True, palette='dark:gray')
g.set_xlabels('')
g.set_ylabels('Mesh density size (equivalent diameter, nm)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()




### PLOT MESH DENSITY 

# scatter 
g = sns.FacetGrid(df_mesh_density, col='time')
g.map_dataframe(sns.stripplot,'cell_type', 'mesh_density', hue='mesh_type', 
                alpha=0.75, dodge=True, palette='muted')
g.set_xlabels('')
g.set_ylabels('Mesh density (%)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()

# box + scatter
g = sns.FacetGrid(df_mesh_density, col='time', col_order=['1min','3min','8min'])
g.map_dataframe(sns.barplot, 'cell_type', 'mesh_density', hue='mesh_type', dodge=True, palette='muted',
                estimator=np.median, errorbar=('pi',50), capsize=0.2)
g.map_dataframe(sns.stripplot,'cell_type', 'mesh_density', hue='mesh_type', 
                alpha=0.5, dodge=True, palette=['black']*2)
g.set_xlabels('')
g.set_ylabels('Mesh density (%)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()





### plot threshold values

# mean 
g = sns.FacetGrid(df_threshold_parameters, col='time', col_order=['1min','3min','8min'])
g.map_dataframe(sns.barplot, 'cell_type', 'mean', hue='mesh_type', dodge=True, palette='muted',
                estimator=np.median, errorbar=('pi',50), capsize=0.2)
g.map_dataframe(sns.stripplot,'cell_type', 'mean', hue='mesh_type', 
                alpha=0.5, dodge=True, palette=['black']*2)
g.set_xlabels('')
g.set_ylabels('Threshold value: mean')
g.add_legend(loc='upper right',title='Mesh type')
plt.show()


# std_dev 
g = sns.FacetGrid(df_threshold_parameters, col='time', col_order=['1min','3min','8min'])
g.map_dataframe(sns.barplot, 'cell_type', 'std_dev', hue='mesh_type', dodge=True, palette='muted',
                estimator=np.median, errorbar=('pi',50), capsize=0.2)
g.map_dataframe(sns.stripplot,'cell_type', 'std_dev', hue='mesh_type', 
                alpha=0.5, dodge=True, palette=['black']*2)
g.set_xlabels('')
g.set_ylabels('Threshold value: standard deviation')
g.add_legend(loc='upper right',title='Mesh type')
plt.show()












#### all cell types


#### PLOT EQUIV DIAMS 

plt.figure(figsize=(8,6))
sns.stripplot(x='cell_type', y='equiv_diameters', data=df_equiv_diams, hue='cell_type', color="gray", edgecolor="black", alpha=0.1, dodge=True)
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


# ANOVA

# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# # Ordinary Least Squares (OLS) model
# model = ols('value ~ C(mesh_type)', data=respd_long[respd_long.variable=='median']).fit()
# anova_table = sm.stats.anova_lm(model, typ=2)
# anova_table

