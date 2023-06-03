""" Post-processing 2D Gaussian fits onto STED images of fluorescent nano beads using Picasso. """

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt 

""" Helpers """

def get_hdf5(file):
    """ Load hdf5 data from Picasso. """
    if "hdf5" in file:
        with h5py.File(file, "r") as hf:
            data = hf['locs'][:]
            header = data.dtype.names
            return data, header 
    else: 
        raise ValueError('File not recognised: {f}'.format(f=file))


def discard_rows(data, cols_subset, rows_discard=None):
    """ Returns a numpy array (with no colnames/data types, i.e. not a structured array)
    which subsets the non-discarded entries from a series of columns 
    """
    subset = []
    for n in range(data.shape[0]):
        if n not in rows_discard:
            subset.append([data[item][n] for item in cols_subset])
    subset = np.array(subset)
    return subset


def get_cols(data, cols_list): 
    # function returns a numpy array which isn't structured and can't be used by rows to discard
    subset_cols = [data[item] for item in cols_list]
    subset_cols = np.array(subset_cols).transpose()
    formats = ['float32']*len(cols_list)
    # format messes up, need to use structured arrays properly
    #dtype_new = np.dtype({'names': cols_list, 'formats': formats}) 
    #subset_cols = subset_cols.astype(dtype_new)
    return subset_cols


""" Post-processing """

# paths
beads_dir = os.path.join(os.getcwd(), 'deconvolution_beads/beads_detected')
filename = 'CARs_11_11_22-0007_locs.hdf5'

# read in data
data, header = get_hdf5(os.path.join(beads_dir, filename))

# args
fields = ['x','y','sx','sy']
to_discard = [2,7,12,15]
keep_potentially = [8,10,11]

# results 
res = discard_rows(data, to_discard, fields)
print(f'Mean sx = {np.mean(res[:,2]):.3f}')
print(f'Mean sy = {np.mean(res[:,3]):.3f}')

np.save(file=os.path.join(os.getcwd(),beads_dir, 'filtered_results_plain.npy'),arr=res)


# save hist figure  
plt.figure()
plt.suptitle(f'Histogram of sx and sy values\nwith mean for n={res.shape[0]} beads')
plt.subplot(1,2,1)
plt.hist(res[:,2], cmap=plt.cm.jet);
sx_mean = np.mean(res[:,2])
plt.axvline(sx_mean, color='darkred')
plt.text(sx_mean,3.05,f'mean={sx_mean:.3f}',color='darkred')
plt.xlabel('sx')
plt.subplot(1,2,2)
plt.hist(res[:,3], cmap=plt.cm.jet);
sy_mean = np.mean(res[:,3])
plt.axvline(sy_mean, color='darkred')
plt.text(sy_mean,3.05,f'mean={sy_mean:.3f}',color='darkred')
plt.xlabel('sy')
plt.savefig(os.path.join(os.getcwd(), beads_dir, 'sx_sy_n=12.png'))
plt.show();


# 2d histograms - a failed attempt 
np.histogram2d(res[:,2], res[:,3])
plt.hist2d(res[:,2], res[:,3], cmap=plt.cm.jet);
plt.show();
