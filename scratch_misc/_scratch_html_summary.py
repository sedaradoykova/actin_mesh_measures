import os, csv, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from meshure.actimg_collection import ActImgCollection, list_files_dir_str

resdir = 'actin_meshwork_analysis/process_data/sample_data/_results_CARs'

filenames, filepaths = list_files_dir_str(resdir)
filenames.keys()
filenames['basal: params']

alldfs = []
for dir, key in zip(('basal','cytosolic'), ('basal: params', 'cytosolic: params')):
    for file in filenames[key]:
        if 'csv' in file:
            path = os.path.join(resdir, dir+'/params/'+file)
            areas = pd.read_csv(path)
            nrep = areas.shape[0]
            with open(path.replace('csv', 'json'), 'r') as f:
                otherparams = json.load(f)        
            df = pd.DataFrame({'filename': np.repeat(otherparams['filename'], nrep),
                               'cell_type': np.repeat(otherparams['cell_type']['type'], nrep),
                               'mesh_type': np.repeat(otherparams['cell_type']['mesh_type'], nrep),
                               'activation_time': np.repeat(otherparams['cell_type']['activation_time'], nrep),
                               'equivalent_diameter_area_um^2': areas['equivalent_diameter_area_um^2'],
                               'area_um^2': areas['area_um^2'],
                               'perimeter_um': areas['perimeter_um']})  
            alldfs.append(df)
len(alldfs)

new_df = pd.concat(alldfs)
new_df.hist();plt.show()


new_df['mesh_type'].unique()
new_df.columns

#new_df = pd.read_csv('actin_meshwork_analysis/process_data/sample_data/_results_CARs/all_params_csv.csv')
new_df = pd.read_csv('actin_meshwork_analysis/process_data/deconv_data/all_params_csv.csv')
new_df['cell_type_dual'] = new_df.apply(lambda x: 'CAR' if 'CAR' in x['cell_type'] else 'Untransduced', axis=1)

new_df.shape

# equivalent diameter area
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


# area
# scatter 
g = sns.FacetGrid(new_df, col='activation_time')
g.map_dataframe(sns.stripplot,'cell_type_dual', 'area_um^2', hue='mesh_type', 
                alpha=0.75, dodge=True, palette='muted')
g.set_xlabels('')
g.set_ylabels('Mesh size (area, $\mu m^2$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()

# box + scatter
g = sns.FacetGrid(new_df, col='activation_time', col_order=['1min','3min','8min'])
g.map_dataframe(sns.barplot, 'cell_type_dual', 'area_um^2', hue='mesh_type', dodge=True, palette='muted',
                estimator=np.median, errorbar=('pi',50), capsize=0.2)
g.map_dataframe(sns.stripplot,'cell_type_dual', 'area_um^2', hue='mesh_type', 
                alpha=0.5, dodge=True, palette='dark:gray')
g.set_xlabels('')
g.set_ylabels('Mesh size (area, $\mu m^2$)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()

# perimeter
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


# surface area 
new_df_surface = new_df[['filename', 'cell_type', 'cell_type_dual', 'mesh_type', 'activation_time','cell_surface_area_um^2', 'mesh_density']].drop_duplicates()


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



# mesh density
# scatter 
g = sns.FacetGrid(new_df_surface, col='activation_time')
g.map_dataframe(sns.stripplot,'cell_type_dual', 'mesh_density', hue='cell_type', 
                alpha=0.75, palette='Accent', dodge=False, linewidth=1)#dodge=True, palette='muted')
g.set_xlabels('')
g.set_ylabels('Actin mesh density (%)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()

# box + scatter
g = sns.FacetGrid(new_df_surface, col='activation_time', col_order=['1min','3min','8min'])
g.map_dataframe(sns.barplot, 'cell_type_dual', 'mesh_density', hue='mesh_type', dodge=True, palette='muted',
                estimator=np.median, errorbar=('pi',50), capsize=0.2)
g.map_dataframe(sns.stripplot,'cell_type_dual', 'mesh_density', hue='cell_type', 
                alpha=0.5, dodge=False, palette='dark:gray') #, dodge=False, palette='Set2')
g.set_xlabels('')
g.set_ylabels('Actin mesh density (%)')
g.add_legend(bbox_to_anchor=(0.9, 0.9),title='Mesh type')
plt.show()


"""
FAILED - include backgroun in labels  
BASAL
3min_FOV1_decon_left
3min_FOV7_decon_top
8min_FOV1_decon_bottom_right
8min_FOV2_decon_bottom
8min_FOV3_decon_top_right
8min_FOV4_decon_bottom
8min_FOV4_decon_top
8min_FOV5_decon_right
8min_FOV6_decon_bottom
8min_FOV6_decon_top
CYTO
1min_FOV8_decon_bottom_left
1min_FOV8_decon_right
3min_FOV5_decon_top             weird artifact 
3min_FOV6_decon_top_right       small problem 
8min_FOV1_decon_bottom_right
8min_FOV2_decon_bottom
8min_FOV3_decon_top_right
8min_FOV4_decon_bottom
8min_FOV5_decon_right
8min_FOV4_decon_bottom
8min_FOV5_decon_right
"""