import os, scipy.ndimage, cv2
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import chain
from meshure.actimg import ActImg, get_ActImg
#from meshure.utils import get_image_stack, list_all_tiffs, get_meta, get_resolution, list_files_dir_str
os.chdir('actin_meshwork_analysis/scratch_misc')
from actin_meshwork_analysis.scratch_misc._scratch_utils import _line_profile_coordinates, plt_threshold_diagnostic
from skimage.measure import profile_line


""" Mesh size and morphological operations. """


data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/sample_data/Untransduced")
os.listdir(data_path)

from PIL import Image


actimg = get_ActImg('3min_Fri_FOV2_decon.tif', data_path) # 1,3 / 6,8
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=[0,60,120],sigma=2,substack=[1,3],visualise=False)

# im = ~actimg.manipulated_stack[0]
# img = Image.fromarray(im)
# img.save('test_image_inv.tiff')

actimg.z_project_min()
actimg.visualise_stack('manipulated',colmap='gray')

linprof_original = profile_line(actimg.manipulated_stack, (0,0), actimg.shape)
linprof = linprof_original[np.argwhere(linprof_original>0)]
plt_threshold_diagnostic(actimg, linprof_original)

actimg.threshold(0.0015) # 0.0234 0.0291 0.0342
actimg.visualise_stack('manipulated',colmap='gray')


from scipy.ndimage import correlate, morphology, label, binary_closing
from skimage import measure
from skimage import morphology as skimorph
img = np.copy(actimg.manipulated_stack)


# closing 
# allows to specify a (square?) structure which is used to close the holes in the image 
closed_image = binary_closing(img, structure = np.ones((15,15))) # compare    5,5   7,7   10,10   15,15   30,30
difference_closed = closed_image-img
labeled_image, num_features = label(closed_image)

# filling 
#filled_image = morphology.binary_fill_holes(img)
filled_image = morphology.binary_fill_holes(img)#,structure=np.ones((5,5)))
difference_filled = filled_image-img


plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title('binary image')
plt.axis('off')
plt.subplot(2,3,2)
plt.imshow(closed_image, cmap='gray')
plt.title('closed image')
plt.axis('off')
plt.subplot(2,3,3)
plt.imshow(difference_closed, cmap='gray')
plt.title('difference image')
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(img, cmap='gray')
plt.title('binary image')
plt.axis('off')
plt.subplot(2,3,5)
plt.imshow(filled_image, cmap='gray')
plt.title('filled image')
plt.axis('off')
plt.subplot(2,3,6)
plt.imshow(difference_filled, cmap='gray')
plt.title('difference image')
plt.axis('off')
plt.show();


# area opening 
# open inverted image and display inverted opened image? 
opened_image = skimorph.area_opening(~img, 600) # area_threshold 
difference = opened_image-img

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('binary image')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(~opened_image, cmap='gray')
plt.title('closed image')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(difference, cmap='gray')
plt.title('difference image')
plt.axis('off')
plt.show();




np.sum(difference_filled)
np.sum(difference_filled)*100 / (difference_filled.shape[0]*difference_filled.shape[1])

labels = measure.label(img)
props = measure.regionprops(labels)

for prop in props:
    print('Label: {} >> Object size: {}'.format(prop.label, prop.area))

props[1].area

plt.imshow(labels, cmap=plt.cm.gnuplot); plt.show();




""" Show diagnostic plot. """

data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/sample_data/CARs")
os.listdir(data_path)

actimg = get_ActImg('3min_FOV3_decon.tif', data_path) # base = [1,4], cyto = [4,7] 
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=[0,60,120],sigma=2,substack=[3,5],visualise=False)
actimg.z_project_min()
#actimg.visualise_stack('manipulated',colmap='gray')
linprof_original = profile_line(actimg.manipulated_stack, (0,0), actimg.shape)
linprof = linprof_original[np.argwhere(linprof_original>0)]

plt_threshold_diagnostic(actimg, linprof_original)



"""Dynamic thresholding. 
    Problems: 
        - cannot use background because 
            - (a) it's the response min proj not the raw image that is being thresholded 
            - (b) it contains negative values (nonsense)
        - even with current ideas it doesn't seem that the process can be done fully automatically... 
    Current ideas: 
        - get diagonal profile and threshold based on it
        - threshold based on 90th percentile of all non-negative response values 
    New ideas: 
        - use something like hysteresis thresholding? (have two thresholds)
        - OR use opening by reconstruction to remove noise from poorly thresholded image 
    Post-meeting ideas: 
        - take line profiles from several places, average out and fit gaussian for dynamic thresholding 
"""

from skimage.measure import profile_line
from matplotlib_scalebar.scalebar import ScaleBar

data_path = os.path.join(os.getcwd(), "../process_data/sample_data/CARs")
os.listdir(data_path)

# Basal [1],[1,3],[3,4],[1,2],[1],[1,4]
# Cytoplasmic [4,6],[6,9],[6,8],[4,6],[3,5],[4,7]

actimg = get_ActImg('3min_FOV3_decon.tif', data_path) # base = [1,4], cyto = [4,7] 
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=[0,60,120],sigma=2,substack=[3,5],visualise=False)
actimg.z_project_min()
#actimg.visualise_stack('manipulated',colmap='gray')
rows, cols = actimg.shape
lineprof_coords = [[(0,0),(rows,cols)], [(rows,0),(0,cols)], 
                   [(int(rows/2),0),(int(rows/2),cols)], [(0,int(cols/2)),(rows,int(cols/2))]]
line_profs = [] 
for start, end in lineprof_coords: 
    line_profs.append(profile_line(actimg.manipulated_stack, start, end, linewidth=5).ravel())

all_profs = np.concatenate(line_profs).ravel()
all_profs.shape
all_profs.sort()
# 2653
#np.savetxt("line_profiles_aggregated.csv", all_profs, delimiter=",")

plt.plot(all_profs, 'o'); plt.show()
plt.hist(all_profs, bins=50); plt.show()
plt.imshow(all_profs.reshape(8,353),cmap='gray');plt.axis('off');plt.show()

import scipy.stats
_, bins, _ = plt.hist(all_profs, 100, density=1, alpha=0.5)
mu, sigma = scipy.stats.norm.fit(all_profs)
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
plt.plot(bins, best_fit_line)
plt.show()
print(f'{mu:.6f}, {sigma:.6f}')
# -0.000876, 0.006132

plt.subplot(1,2,1)
plt.imshow(actimg.manipulated_stack,cmap='gray')
scalebar = ScaleBar(actimg.resolution, 'nm', box_color='None', color='black', location='upper left') 
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.title('steerable Gaussian filter resposne')
for (r1,r2),(c1,c2) in lineprof_coords: 
    plt.plot((r1,c1),(r2,c2), color='black')
plt.subplot(1,2,2)
_, bins, _ = plt.hist(all_profs, 100, density=1, alpha=0.95, color='#A8A8A8')
plt.plot(bins, best_fit_line, color='black')
plt.ylabel('Count')
plt.xlabel('Pixel intensity value')
plt.title('aggregated line profiles\nafter steerable Gaussian filter')
plt.show();



from scipy.optimize import curve_fit
nbins = 400

_, bins, _ = plt.hist(all_profs, nbins, density=1, alpha=0.8)
mu, sigma_scipy = scipy.stats.norm.fit(all_profs)
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma_scipy)
plt.plot(bins, best_fit_line)
plt.show()


def gaussian(x, mu, sigma, a, c):
    y = a*np.exp(-(x-mu)**2/(2*sigma**2)) + c
    return y 


# scipy fit
ns, bins, _ = plt.hist(all_profs, nbins, density=1, color='#A8A8A8') # alpha=0.5
binned_prof = (bins+(bins[1] - bins[0])/2)[:-1]
mean = sum(ns*binned_prof)/sum(ns)                  
sigma = sum(ns*(binned_prof-mean)**2)/sum(ns) 

#ns[np.argmax(ns)] = 0.3*np.max(ns)
plt.plot(ns); plt.show()
fit_params, fit_covs = curve_fit(gaussian, binned_prof, ns, p0=[mean,sigma,500,0])
                                            # bounds = ([all_profs[0], 0, 0, 0], # min mu, sigma, a, c 
                                            #     [1, np.inf, np.inf, np.inf])) # max mu, sigma, a, 
fit_y = gaussian(binned_prof, *fit_params)
# full width at half maximum
fwhm = 2*np.sqrt(2*np.log(2))*fit_params[1]

plt.plot(binned_prof, ns, color='black', label='binned points')
plt.legend(loc='upper left')
plt.show()


plt.hist(all_profs, nbins, density=True, color='#A8A8A8') # alpha=0.5
plt.plot(binned_prof, ns,':.',color='black', alpha=0.75, label='binned points')
plt.plot(binned_prof, fit_y,'--',color = 'red', alpha=0.75, label='gauss fit')
#plt.xlim(-.02,.02)
plt.legend(loc='upper left')
plt.show()
print(f'\n                 MU              SIGMA'+
      f'\nmy fitter:       {fit_params[0]:.6f}       {fit_params[1]:.6f}'+
      f'\nscipy.fit.norm:  {mu:.6f}       {sigma:.6f}\n')


plt.subplot(1,2,1)
plt.imshow(actimg.manipulated_stack,cmap='gray')
scalebar = ScaleBar(actimg.resolution, 'nm', box_color='None', color='black', location='upper left') 
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.title('steerable Gaussian filter resposne')
for (r1,r2),(c1,c2) in lineprof_coords: 
    plt.plot((r1,c1),(r2,c2), color='black')
plt.subplot(1,2,2)
_, bins, _ = plt.hist(all_profs, 30, density=1, alpha=0.95, color='#A8A8A8')
plt.plot(bins, fit_y, color='black')
plt.ylabel('Count')
plt.xlabel('Pixel intensity value')
plt.title('aggregated line profiles\nafter steerable Gaussian filter')
plt.show();

plt.close();


import scipy.signal as signal
width, width_height, int_ptsL, int_ptsR = signal.peak_widths(ns, signal.find_peaks(ns, 0.3*(len(ns)))[0]) # this measure depends on bin width 
plt.plot(ns);plt.show()
width[0]*(binned_prof[1] - binned_prof[0])
# 0.00265017 vs fit 0.00101703642
# 0.00128143 vs fit 0.000178805226
fit_params
ns[76:80]

np.allclose(bins[1] - bins[0], binned_prof[1]-binned_prof[0])


# fit to kernel density plot - not as brilliant and promising as once hoped 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
ns, bins, _ = plt.hist(all_profs, nbins, density=1, color='#A8A8A8') # alpha=0.5
binned_prof = (bins+(bins[1] - bins[0])/2)[:-1]

density = gaussian_kde(ns)
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.plot(binned_prof,density(binned_prof))
plt.show()

pd.DataFrame({'counts':ns, 'int': binned_prof}).counts.plot(kind='density'); plt.show()

"""
fig = plt.figure()
x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),nbins)
plt.plot(x_hist_2,gaussian(x_hist_2,*param_optimised),'r--',label='Gaussian fit')
plt.legend()
# Normalise the histogram values
weights = np.ones_like(all_profs) / len(all_profs)
plt.hist(all_profs, weights=weights, bins=nbins)
plt.show()
"""


img = actimg.manipulated_stack.copy()
actimg.nuke()
actimg.z_project_max([3,5])
max_proj = actimg.manipulated_stack.copy()
mu, sigma = fit_params[0:2]
thresh_ways = ['max_proj', 'img < mu', 'img < mu-0.5*sigma', 'img < mu-1*sigma', 'img < mu-1.5*sigma', 'img < mu-2*sigma', 
               'img', 'img > mu', 'img > mu+0.5*sigma', 'img > mu+1*sigma', 'img > mu+1.5*sigma', 'img > mu+2*sigma']
plt_rows = 2
for n, exp in enumerate(thresh_ways):
    plt.subplot(plt_rows, int(len(thresh_ways)/plt_rows), n+1)
    thresh = eval(exp)
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')
    plt.title(exp)
plt.show()


np.percentile(linprof, 99)

coords = _line_profile_coordinates((0,0), actimg.shape)
coords.shape

order, mode, cval = 1, 'reflect', 0.0

pixels = scipy.ndimage.map_coordinates(actimg.manipulated_stack, coords, prefilter=order > 1,
                                     order=order, mode=mode, cval=cval)

pixels[pixels <= 1e-10] = 0
resized_im = cv2.resize(np.transpose(pixels), (pixels.shape[0], 100))
plt.imshow(resized_im, cmap='gray')
plt.axis('off')
plt.show()




fig = plt.figure(figsize=(8, 7))
fig.add_subplot(221)
# plt.subplot(2,2,1)
plt.imshow(actimg.manipulated_stack,cmap='gray')
scalebar = ScaleBar(actimg.resolution, 'nm', box_color='None', color='black', location='upper left') 
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.plot((0,actimg.shape[0]),(0,actimg.shape[1]), color='black')
plt.title('(A) Minimum projection of\nsteerable Gaussian filter response')
# ax = plt.subplot(2,2,3)
ax = fig.add_subplot(223)
plt.imshow(resized_im, cmap='gray', extent=[0, resized_im.shape[1], 300, 350])
plt.plot(linprof_original*10000, color='black')
plt.ylim(0,1.2*np.max(linprof_original*10000))
oticks = ax.get_yticks()
nlabels = np.round(np.linspace(0,np.max(linprof_original), len(oticks[:-1])), 4)
plt.yticks(ticks=oticks[:-1], labels=nlabels)
plt.xlabel('Pixel location on line')
plt.ylabel('Pixel intensity value')
plt.title('(B) Line profile of steerable\nGaussian filter response')
# plt.subplot(2,2,2)
fig.add_subplot(122)
plt.hist(linprof,color='#545454')
plt.xlabel('Pixel intensity value')
plt.ylabel('Count')
plt.title('(C) Histogram of\nnon-negative intensity values')
#plt.tight_layout()
plt.show();
#fig.savefig('actin_meshwork_analysis/meshwork/test_data/figs/line_intensity.png',transparent=True)


plt.close();


"""
- define what the focal plane is (done by Sabrina and Olivia, see excel.)
    - basal and cytoplasmic meshes
- more filling - find outline of cell and calculate 
    - maybe sequential with small steps? find outline
- size of mesh area is more important than percentage cell covered by mesh
"""

# im = get_image_stack(os.path.join(os.getcwd(), 'meshwork/test_data', '7min_Dual_fov2_deconv_1-3.tiff'), verbose=False)
# im[0][0].shape

# len(im)
# im[1]



""" Working out the mesh size (old). """

actimg = get_ActImg('7min_Dual_fov2_deconv_1-3.tiff', os.path.join(os.getcwd(), 'meshwork/test_data'))
actimg.normalise()
actimg.z_project_min()
actimg.visualise('manipulated')
actimg.threshold(0.03)
actimg.visualise('manipulated')

from scipy.ndimage import correlate, morphology, label, binary_closing
from skimage import measure
img = np.copy(actimg.manipulated_stack)

closed_image = binary_closing(img, structure = np.ones((50,50)))
plt.imshow(closed_image, cmap='inferno'); plt.show();
difference2 = closed_image-img
plt.imshow(difference2, cmap='inferno'); plt.show();
labeled_image, num_features = label(closed_image)
print(np.bincount(labeled_image.ravel())[1:])

filled_image = morphology.binary_fill_holes(img)
plt.imshow(filled_image, cmap='inferno'); plt.show();
difference = filled_image-img
plt.imshow(difference, cmap='inferno'); plt.show();

np.sum(difference)
np.sum(difference)*100 / (difference.shape[0]*difference.shape[1])

labels = measure.label(img)
props = measure.regionprops(labels)

for prop in props:
    print('Label: {} >> Object size: {}'.format(prop.label, prop.area))

props[1].area

plt.imshow(labels, cmap=plt.cm.gnuplot); plt.show();

# actimg = get_ActImg('7min_Dual_fov2_deconv_1-3.tiff', os.path.join(os.getcwd(), 'meshwork/test_data'))

# actimg.visualise('original', 1)
# actimg.normalise()
# actimg.z_project_min()
# actimg.threshold(0.015)
# actimg.visualise('manipulated')

# actimg.nuke()
# actimg.normalise()
# actimg.z_project_max()
# actimg.visualise('manipulated')


# meta = get_meta(os.path.join(os.getcwd(), 'test_data', 'CARs_11_11_22.lif - Fri_Beads.tif'))
# get_resolution(meta, True)
# meta.keys()

# meta['XResolution']
# meta['ImageDescription']

# 32903098/1000000




def z_proj_im():
    ones = np.ones(5)
    im = np.array([
        [*ones, *ones],[*ones, *ones],[*ones, *ones],
        [*ones*9, *ones*9],[*ones*5, *ones*5],[*ones*5, *ones*5],[*ones*9, *ones*9],
        [*ones, *ones],[*ones, *ones],[*ones, *ones]
        ])
    return np.array([im, np.transpose(im), np.arange(100).reshape(10,10)])


# pipeline works?
actimg = ActImg(z_proj_im(), 'test.tiff', (10,10), 3, False, 20) 
#actimg.visualise_stack()
actimg.image_stack
actimg.normalise()
#actimg.visualise('manipulated')
actimg.manipulated_stack
actimg.z_project_max([0,1])
actimg.visualise('manipulated')

actimg.steerable_gauss_2order(None,2,360,True)
actimg.visualise_stack('manipulated', colmap='gray')

# bug in z-projection due to complex numbers (after steerable filter)
# what does np.real(x) do?? 
# try np.max(IM, axis=0)
actimg.z_project_min([0,1])
actimg.visualise('manipulated', colmap='gray')


# THERE MUST BE A BUG 
actimg = ActImg(z_proj_im(), 'test.tiff', (10,10), 3, False, 20) 
actimg.normalise()
actimg.steerable_gauss_2order([0,2],2,360,True)
actimg = ActImg(z_proj_im(), 'test.tiff', (10,10), 3, False, 20) 
actimg.normalise()
actimg.steerable_gauss_2order([1,2],2,360,True)
actimg = ActImg(z_proj_im(), 'test.tiff', (10,10), 3, False, 20) 
actimg.normalise()
actimg.steerable_gauss_2order([2,2],2,360,True)



# import numpy as np 
# import matplotlib.pyplot as plt
# from meshwork.meshwork import ActImg, get_ActImg
# from meshwork.utils import get_image_stack, list_all_tiffs, get_meta, get_resolution, list_files_dir_str

# def z_proj_im():
#     ones = np.ones(5)
#     im = np.array([
# [*ones, *ones],[*ones, *ones],[*ones, *ones],[*ones*9, *ones*9],
# [*ones*5, *ones*5],[*ones*5, *ones*5],[*ones*9, *ones*9],[*ones, *ones],[*ones, *ones],[*ones, *ones]
# ])
#     return np.array([im, np.transpose(im)])


# actimg = ActImg(z_proj_im(), 'test.tiff', (10,10), 2, False, 20) 

# import os
# actimg = get_ActImg('7min_Dual_fov2_deconv_1-3.tiff', os.path.join(os.getcwd(), 'meshwork/test_data'))
# actimg.visualise_stack()
# actimg.normalise()
# actimg.z_project_max([1,2])
# actimg.visualise_stack('manipulated')

from scipy import io
avg_out = 'meshwork/steerable_gaussian/matlab_files/steerable_filters/steerable filters/average.mat'
mat = io.loadmat(avg_out)['Average']

plt.imshow(mat, cmap='gray'); plt.show();
np.max(mat)



impath = 'meshwork/steerable_gaussian/matlab_files/steerable_filters/steerable filters/'
actimg = get_ActImg('testUTR_3minTHUfov1_04.tif', os.path.join(os.getcwd(), impath))
#actimg.visualise()
actimg.normalise()

actimg.steerable_gauss_2order(None,2,30,True)


actimg.visualise_stack('manipulated', colmap='gray')


np.histogram(actimg.image_stack)
plt.imshow(actimg.image_stack[0], cmap='gray', vmax=5e4);plt.show()


sigma, theta = 2, 30

theta = np.deg2rad(theta)

Wx = np.floor((8/2)*sigma)
Wx = Wx if Wx >= 1 else 1

x = np.arange(-Wx,Wx+1) ### tricky part because this chooses the output size??? 
xx,yy = np.meshgrid(x,x)


g0 = np.array(np.exp(-(xx**2+yy**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)))
G2a = np.array(-g0/sigma**2+g0*xx**2/sigma**4)
G2b =  np.array(g0*xx*yy/sigma**4)
G2c = np.array(-g0/sigma**2+g0*yy**2/sigma**4)
oriented_filter = (np.cos(theta))**2*G2a + np.sin(theta)**2*G2c - 2*np.cos(theta)*np.sin(theta)*G2b



for i, (im, title) in enumerate(zip([g0, G2a, G2b, G2c, oriented_filter], ['g0', 'G2a', 'G2b', 'G2c', 'oriented_filter'])):
    plt.subplot(1,5,i+1)
    plt.imshow(im, cmap='gray'); 
    plt.axis('off')
    plt.title(title)
plt.show();

gs = [g0, G2a, G2b, G2c]

min(list(map(np.min, gs)))
max(list(map(np.max, gs)))

# there are negative values in the gs which explains why we get negative values in the indecies 
# python doesn't support power operations for negative fractions
# hence the error
# the solution is to convert to complex numbers.... 






####################### GAUSS DEBUG

import os
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import chain
from meshure.actimg import ActImg, get_ActImg
#from meshwork.utils import get_image_stack, list_all_tiffs, get_meta, get_resolution, list_files_dir_str


impath = 'meshwork/'
actimg = get_ActImg('testUTR_3miFRIifov1_01.tif', impath)
actimg.manipulated_stack = actimg.image_stack.copy()

#actimg.visualise()


actimg.steerable_gauss_2order(None,2,90,True)
actimg.visualise(imtype='manipulated')

actimg = get_ActImg('testUTR_3miFRIifov1_01.tif', impath)
actimg.normalise()

#res_fig, res_stacks = [], [] 
res = [0]*6

for n, angle in enumerate(np.arange(0,360,60)):
    res[n] = actimg.steerable_gauss_2order(substacks=None, sigma=2, theta=angle, visualise=True)
    actimg.nuke()
    actimg.normalise()

mean = np.mean(np.asarray(res),0)
plt.imshow(mean, cmap='gray'); plt.show()


from scipy import io
j2_out = 'meshwork/steerable_gaussian/matlab_files/steerable_filters/steerable filters/average.mat'
mat = io.loadmat(j2_out)['J2']
j2_mean = np.mean(mat, 2)

plt.imshow(j2_mean, cmap='gray'); plt.show();


# shapes match 
j2_mean.shape == mean.shape

np.allclose(j2_mean, mean, atol=1e-20)

minmin = np.min([np.min(j2_mean), np.min(mean)])
maxmax = np.max([np.max(j2_mean), np.max(mean)])

plt.figure(figsize=(10,4))
plt.subplots_adjust(hspace=0.01, vspace=0, top=0)
plt.subplot(1,3,1)
plt.imshow(j2_mean, cmap='gray', vmax=maxmax, vmin=minmin)
plt.title('matlab output')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(mean, cmap='gray', vmax=maxmax, vmin=minmin)
plt.title('python output')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(mean-j2_mean,cmap='gray', vmax=maxmax, vmin=minmin)
plt.title('difference')
plt.axis('off')
plt.suptitle('comparing output for averaging over $\\theta$=[0,60,120,180,240,300]\nnp.allclose(matlab_mean, pyhton_mean, atol=1e-8)==True', fontsize=15)
plt.show();
