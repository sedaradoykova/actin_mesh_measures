import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from meshure.actimg_collection import ActImgCollection, list_files_dir_str

# data paths 
data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/deconv_data")
focal_plane_filename = os.path.join(data_path, 'all_data_focal_planes.csv')
only_subdirs = ['all_Untransduced','all_CARs']

# analysis parameters
theta_x21 = np.linspace(0,180,21)
parameters = [None, None, theta_x21, 2, None, None]


actimg_collection = ActImgCollection(root_path=data_path)
actimg_collection.print_summary(only_subdirs)
actimg_collection.get_focal_planes(focal_plane_filename)
actimg_collection.only_subdirs = only_subdirs
actimg_collection.parametrise_pipeline(*parameters)


actimg_collection.run_analysis(visualise_as_html=True, return_parameters=False, save_as_single_csv=True)
## 06/04 Analysis completed in 00:18:44.
## 07/04 Analysis completed in 00:33:47.
actimg_collection.failed_segmentations

"""
UNTR VS CAR 

Untransduced                : 1min  : 16
Untransduced                : 3min  : 16
Untransduced                : 8min  : 16

CARs                        : 1min  : 21
CARs                        : 3min  : 20
CARs                        : 8min  : 18

Untransduced                : total : 48
CARs                        : total : 59

Total files                 : 107

ALL TYPES
Untransduced        46
CAR_dual            28
CAR_antiCD19        13
CAR_antiCD22        7
"""

""" Visualise results. """

# load data 
new_df = pd.read_csv('actin_meshwork_analysis/process_data/deconv_data/all_params_csv.csv')
new_df['cell_type_dual'] = new_df.apply(lambda x: 'CAR' if 'CAR' in x['cell_type'] else 'Untransduced', axis=1)
new_df.shape

# drop duplicates for surface area and mesh density % 
new_df_surface = new_df[['filename', 'cell_type', 'cell_type_dual', 'mesh_type', 'activation_time','cell_surface_area_um^2', 'mesh_density']].drop_duplicates()
new_df_surface.shape

# drop unwwanted cases
for filename, celltype, meshtype in [['8min_UNT_FOV3_decon.tif', 'Untransduced', 'Cytosolic'],
                                     ['8min_UNT_FOV6_decon.tif', 'Untransduced', 'Cytosolic'],
                                     ['8min_UNT_FOV7_decon.tif', 'Untransduced', 'Cytosolic'],
                                     ['3min_UNT_FOV1_decon.tif', 'Untransduced', 'Cytosolic'],
                                     ['3min_UNT_FOV1_decon.tif', 'Untransduced', 'Basal'],
                                     ['1min_FOV8_decon_right.tif', 'CAR', 'Cytosolic'],
                                     ['3min_FOV1_decon_left.tif', 'CAR', 'Basal'],
                                     ['3min_FOV4_decon_top_right.tif', 'CAR', 'Cytosolic'],
                                     ['3min_FOV5_decon_top.tif', 'CAR', 'Cytosolic'], 
                                     ['8min_FOV6_decon_top.tif','CAR', 'Cytosolic'],
                                     ['8min_FOV6_decon_top.tif','CAR', 'Basal']]:
    new_df_surface.drop(new_df_surface.loc[(new_df_surface['filename']==filename) &
                                           (new_df_surface['cell_type_dual']==celltype) &
                                           (new_df_surface['mesh_type']==meshtype)].index, inplace=True)
new_df_surface.shape

new_df_surface.to_csv('all_data_summary.csv',header=True,index=False)

#new_df_surface.set_index(new_df_surface['filename'], inplace=True)

for file, ctype in zip(new_df_surface['filename'], new_df_surface['cell_type']):
    for stat in ['equivalent_diameter_area_um^2', 'area_um^2', 'perimeter_um']:
        ind = new_df_surface[(new_df_surface['filename']==file) & (new_df_surface['cell_type']==ctype)].index
        new_df_surface.loc[ind,f'{stat}_mean'] = new_df[(new_df['filename']==file) & (new_df['cell_type']==ctype)][stat].mean()
        new_df_surface.loc[ind,f'{stat}_median'] = new_df[(new_df['filename']==file) & (new_df['cell_type']==ctype)][stat].median()



new_df_surface.head()



#### cell surface area 
# scatter 
g = sns.FacetGrid(new_df_surface, col='activation_time')
g.map_dataframe(sns.stripplot,'cell_type_dual', 'cell_surface_area_um^2', hue='mesh_type', 
                alpha=0.75, dodge=True, palette='muted')
g.set_xlabels('')
g.set_ylabels('Cell surface area ($\mu m^2$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()

# box + scatter
g = sns.FacetGrid(new_df_surface, col='activation_time', col_order=['1min','3min','8min'])
g.map_dataframe(sns.barplot, 'cell_type_dual', 'cell_surface_area_um^2', hue='mesh_type', dodge=True, palette='muted',
                estimator=np.median, errorbar=('pi',50), capsize=0.2)
g.map_dataframe(sns.stripplot,'cell_type_dual', 'cell_surface_area_um^2', hue='mesh_type', 
                alpha=0.5, dodge=True, palette='dark:gray')
g.set_xlabels('')
g.set_ylabels('Cell surface area ($\mu m^2$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()



#### mesh density
# scatter 
sns.set_style("whitegrid")
g = sns.FacetGrid(new_df_surface, col='activation_time', row='mesh_type')
g.map_dataframe(sns.stripplot,'cell_type_dual', 'mesh_density', hue='cell_type',
                alpha=0.75, palette='Accent', dodge=False, linewidth=1)#dodge=True, palette='muted')
g.set_xlabels('')
g.set_ylabels('Actin mesh density (%)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()


# box + scatter
g = sns.FacetGrid(new_df_surface, col='activation_time')
g.map_dataframe(sns.barplot, 'cell_type_dual', 'mesh_density', hue='mesh_type', dodge=True, palette='muted',
                estimator=np.median, errorbar=('pi',50), capsize=0.2)
g.map_dataframe(sns.stripplot,'cell_type_dual', 'mesh_density', hue='mesh_type', 
                alpha=0.5, dodge=True, palette='dark:gray') #, dodge=False, palette='Set2')
g.set_xlabels('')
g.set_ylabels('Actin mesh density (%)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()



new_df_surface.columns
#### hole area
# scatter 
g = sns.FacetGrid(new_df_surface, col='activation_time')
g.map_dataframe(sns.stripplot,'cell_type_dual', 'area_um^2_median', hue='mesh_type', 
                alpha=0.75, dodge=True, palette='muted')
g.set_xlabels('')
g.set_ylabels('Median mesh size (hole area, $\mu m^2$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()

# box + scatter
g = sns.FacetGrid(new_df_surface, col='activation_time')
g.map_dataframe(sns.barplot, 'cell_type_dual', 'area_um^2_median', hue='mesh_type', dodge=True, palette='muted',
                estimator=np.median, errorbar=('pi',50), capsize=0.2)
g.map_dataframe(sns.stripplot,'cell_type_dual', 'area_um^2_median', hue='mesh_type', 
                alpha=0.5, dodge=True, palette='dark:gray')
g.set_xlabels('')
g.set_ylabels('Median mesh size (hole area, $\mu m^2$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()



#### hole equivalent diameter area
# scatter 
g = sns.FacetGrid(new_df_surface, col='activation_time')
g.map_dataframe(sns.stripplot,'cell_type_dual', 'equivalent_diameter_area_um^2_median', hue='mesh_type', 
                alpha=0.75, dodge=True, palette='muted')
g.set_xlabels('')
g.set_ylabels('Mesh size (median equivalent diameter area, $\mu m^2$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()

# box + scatter
g = sns.FacetGrid(new_df_surface, col='activation_time')
g.map_dataframe(sns.barplot, 'cell_type_dual', 'equivalent_diameter_area_um^2_median', hue='mesh_type', dodge=True, palette='muted',
                estimator=np.median, errorbar=('pi',50), capsize=0.2)
g.map_dataframe(sns.stripplot,'cell_type_dual', 'equivalent_diameter_area_um^2_median', hue='mesh_type', 
                alpha=0.5, dodge=True, palette='dark:gray')
g.set_xlabels('')
g.set_ylabels('Mesh size (median equivalent diameter area, $\mu m^2$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()



#### hole perimeter
# scatter 
g = sns.FacetGrid(new_df_surface, col='activation_time')
g.map_dataframe(sns.stripplot,'cell_type_dual', 'perimeter_um_median', hue='mesh_type', 
                alpha=0.75, dodge=True, palette='muted')
g.set_xlabels('')
g.set_ylabels('Mesh size (median hole perimeter, $\mu m$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()

# box + scatter
g = sns.FacetGrid(new_df_surface, col='activation_time', col_order=['1min','3min','8min'])
g.map_dataframe(sns.barplot, 'cell_type_dual', 'perimeter_um_median', hue='mesh_type', dodge=True, palette='muted',
                estimator=np.median, errorbar=('pi',50), capsize=0.2)
g.map_dataframe(sns.stripplot,'cell_type_dual', 'perimeter_um_median', hue='mesh_type', 
                alpha=0.5, dodge=True, palette='dark:gray')
g.set_xlabels('')
g.set_ylabels('Mesh size (median hole perimeter, $\mu m$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()



############################################################


#### mesh density
# scatter 
g = sns.FacetGrid(new_df_surface, col='activation_time')
g.map_dataframe(sns.stripplot,'cell_type_dual', 'mesh_density', hue='mesh_type',
                alpha=0.75, palette='Accent', dodge=True, linewidth=1)#dodge=True, palette='muted')
g.set_xlabels('')
g.set_ylabels('Actin mesh density (%)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()

# box + scatter
g = sns.FacetGrid(new_df_surface, col='activation_time')
g.map_dataframe(sns.barplot, 'cell_type_dual', 'mesh_density', hue='mesh_type', dodge=True, palette='muted',
                estimator=np.median, errorbar=('pi',50), capsize=0.2)
g.map_dataframe(sns.stripplot,'cell_type_dual', 'mesh_density', hue='cell_type', 
                alpha=0.5, dodge=False, palette='dark:gray') #, dodge=False, palette='Set2')
g.set_xlabels('')
g.set_ylabels('Actin mesh density (%)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()



#### hole area
# scatter 
g = sns.FacetGrid(new_df, col='activation_time')
g.map_dataframe(sns.stripplot,'cell_type_dual', 'area_um^2', hue='mesh_type', 
                alpha=0.75, dodge=True, palette='muted')
g.set_xlabels('')
g.set_ylabels('Mesh size (area, $\mu m^2$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()

# box + scatter
g = sns.FacetGrid(new_df, col='activation_time')
g.map_dataframe(sns.barplot, 'cell_type_dual', 'area_um^2', hue='mesh_type', dodge=True, palette='muted',
                estimator=np.median, errorbar=('pi',50), capsize=0.2)
g.map_dataframe(sns.stripplot,'cell_type_dual', 'area_um^2', hue='mesh_type', 
                alpha=0.5, dodge=True, palette='dark:gray')
g.set_xlabels('')
g.set_ylabels('Mesh size (area, $\mu m^2$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()



#### hole equivalent diameter area
# scatter 
g = sns.FacetGrid(new_df, col='activation_time')
g.map_dataframe(sns.stripplot,'cell_type_dual', 'equivalent_diameter_area_um^2', hue='mesh_type', 
                alpha=0.75, dodge=True, palette='muted')
g.set_xlabels('')
g.set_ylabels('Mesh size (equivalent diameter area, $\mu m^2$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()

# box + scatter
g = sns.FacetGrid(new_df, col='activation_time', col_order=['1min','3min','8min'])
g.map_dataframe(sns.barplot, 'cell_type_dual', 'equivalent_diameter_area_um^2', hue='mesh_type', dodge=True, palette='muted',
                estimator=np.median, errorbar=('pi',50), capsize=0.2)
g.map_dataframe(sns.stripplot,'cell_type_dual', 'equivalent_diameter_area_um^2', hue='mesh_type', 
                alpha=0.5, dodge=True, palette='dark:gray')
g.set_xlabels('')
g.set_ylabels('Mesh size (equivalent diameter area, $\mu m^2$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()



#### hole perimeter
# scatter 
g = sns.FacetGrid(new_df, col='activation_time')
g.map_dataframe(sns.stripplot,'cell_type_dual', 'perimeter_um', hue='mesh_type', 
                alpha=0.75, dodge=True, palette='muted')
g.set_xlabels('')
g.set_ylabels('Mesh size (hole perimeter, $\mu m$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()

# box + scatter
g = sns.FacetGrid(new_df, col='activation_time', col_order=['1min','3min','8min'])
g.map_dataframe(sns.barplot, 'cell_type_dual', 'perimeter_um', hue='mesh_type', dodge=True, palette='muted',
                estimator=np.median, errorbar=('pi',50), capsize=0.2)
g.map_dataframe(sns.stripplot,'cell_type_dual', 'perimeter_um', hue='mesh_type', 
                alpha=0.5, dodge=True, palette='dark:gray')
g.set_xlabels('')
g.set_ylabels('Mesh size (hole perimeter, $\mu m$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()
