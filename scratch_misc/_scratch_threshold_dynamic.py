import os
import numpy as np 
import matplotlib.pyplot as plt
from meshure.actinimg import get_ActinImg

data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/sample_data/CARs")
os.listdir(data_path)
# Untr 3min_Fri_FOV2_decon.tif ## 1,3 and 6,8
# CARs 3min_FOV3_decon.tif ## 1 and ??? 
actimg = get_ActinImg('3min_FOV3_decon.tif', data_path) 
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=[0,60,120],sigma=2,substack=[1],visualise=False)
actimg.z_project_min()

#bins = 20,30,50,100,150,200,250,300,350    # fails at 20, 300
#bins = 20,30,40,50,100,150,200,250,300,350,400
bins = np.arange(10,401,10)
bins_res_mu, bins_res_sigma = [None]*len(bins), [None]*len(bins) 
fails = []
for n, bin in enumerate(bins):
    try: 
        bins_res_mu[n], bins_res_sigma[n] = actimg.threshold_dynamic(None,bin,'curve_fit',vis=False)
    except:
        bins_res_mu[n], bins_res_sigma[n] = np.nan, np.nan
        fails.append(bin)
fails

mu_mean, mu_median = np.nanmean(bins_res_mu), np.nanmedian(bins_res_mu)
sigma_mean, sigma_median = np.nanmean(bins_res_sigma), np.nanmedian(bins_res_sigma)


plt.subplot(1,2,1)
plt.scatter(bins, bins_res_mu, s=15) # does not change
plt.scatter(fails, [0]*len(fails), marker='x', color='black', s=15)
plt.xlabel('n bins'); plt.ylabel('mu'); 
#plt.ylim(-0.001,0.001) for CAR
plt.hlines(y=0,xmin=20,xmax=400, linestyles='--', colors='red')
plt.hlines(y=mu_mean,xmin=20,xmax=400, linestyles='dotted', colors='darkblue', label=f'mean {mu_mean:.5f}')
plt.hlines(y=mu_median,xmin=20,xmax=400, linestyles='dotted', colors='darkred', label=f'median {mu_median:.5f}')
plt.legend()
plt.subplot(1,2,2)
plt.scatter(bins, bins_res_sigma, s=15) 
plt.scatter(fails, [0]*len(fails), marker='x', color='black', s=15)
plt.xlabel('n bins'); plt.ylabel('sigma'); 
#plt.ylim(-0.001,0.01) for CAR
plt.hlines(y=0,xmin=20,xmax=400, linestyles='--', colors='red')
plt.hlines(y=sigma_mean,xmin=20,xmax=400, linestyles='dotted', colors='darkblue', label=f'mean {sigma_mean:.5f}')
plt.hlines(y=sigma_median,xmin=20,xmax=400, linestyles='dotted', colors='darkred', label=f'median {sigma_median:.5f}')
plt.legend()
plt.suptitle('Mu and sigma values against number of bins used for fit')
plt.show()


actimg = get_ActinImg('3min_FOV3_decon.tif', data_path) # base = [1,4], cyto = [4,7] 
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=[0,60,120],sigma=2,substack=[3,5],visualise=False)
actimg.z_project_min()


mu, sigma = actimg.threshold_dynamic(None,10,'curve_fit',vis=True) # scipy.stats.norm.fit


img = actimg.manipulated_stack.copy()
actimg.nuke()
actimg.z_project_max([3,5])
max_proj = actimg.manipulated_stack.copy()

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

exp = 'img < mu-0.5*sigma'
thresh = eval(exp)
plt.imshow(thresh, cmap='gray')
plt.axis('off')
plt.title(exp)
plt.show()



from skimage.measure import profile_line


data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/sample_data/CARs")
os.listdir(data_path)
# Untr 3min_Fri_FOV2_decon.tif ## 1,3 and 6,8
# CARs 3min_FOV3_decon.tif ## 1 and ??? 
actimg = get_ActinImg('3min_FOV3_decon.tif', data_path) 
actimg.normalise()
actimg.steerable_gauss_2order_thetas(thetas=[0,60,120],sigma=2,substack=[1],visualise=False)
actimg.z_project_min()


rows, cols = actimg.shape
lineprof_coords = [[(0,0),(rows,cols)], [(rows,0),(0,cols)], 
                   [(int(rows/2),0),(int(rows/2),cols)], [(0,int(cols/2)),(rows,int(cols/2))]]
line_profs = [] 
for start, end in lineprof_coords: 
    line_profs.append(profile_line(actimg.manipulated_stack, start, end, linewidth=5).ravel())

all_profs = np.concatenate(line_profs).ravel()
all_profs.shape
all_profs.sort()

nbins = 400
ns, bins, _ = plt.hist(all_profs, nbins, density=1, color='#A8A8A8') # alpha=0.5
binned_prof = (bins+(bins[1] - bins[0])/2)[:-1]
# of histogram
# mu = sum(ns*binned_prof)/sum(ns)                  
# sigma = sum(ns*(binned_prof-mu)**2)/sum(ns)

# of data 
mu, sigma = np.mean(all_profs), np.std(all_profs)


img = actimg.manipulated_stack.copy()
actimg.nuke()
actimg.z_project_max([3,5])
max_proj = actimg.manipulated_stack.copy()

thresh_ways = ['max_proj', 'img < mu', 'img < mu-0.5*sigma', 'img < mu-1*sigma', 'img < mu-1.5*sigma', 'img < mu-2*sigma', 
               'img', 'img > mu', 'img > mu+0.5*sigma', 'img > mu+1*sigma', 'img > mu+1.5*sigma', 'img > mu+2*sigma']
# thresh_ways = ['max_proj', 'img < mu', 'img < mu-2*sigma', 'img < mu-3*sigma', 'img < mu-5*sigma', 'img < mu-10*sigma', 
#                'img', 'img > mu', 'img > mu+sigma', 'img > mu+2*sigma', 'img > mu+3.5*sigma', 'img > mu+3*sigma']


plt_rows = 2
for n, exp in enumerate(thresh_ways):
    plt.subplot(plt_rows, int(len(thresh_ways)/plt_rows), n+1)
    thresh = eval(exp)
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')
    plt.title(exp)
plt.suptitle(f'Threshold for mu = {mu:.5f} and sigma = {sigma:.5f}')
plt.show()





### TRY DIFFERENT MORPHOLOGICAL OPERATIONS 
# it appears that closing is indeed the best operation
# the problem remains: what structure to choose.... 

from scipy.ndimage import correlate, morphology, label, binary_closing
from skimage import measure
from skimage import morphology as skimorph
#img = np.copy(actimg.manipulated_stack)
img = np.copy(thresh)

# closing 
# allows to specify a (square?) structure which is used to close the holes in the image 
# closing = dilation followed by erosion with the same structuring element 
closed_image = binary_closing(img, structure = np.ones((15,15))) # compare    5,5   7,7   10,10   15,15   30,30
difference_closed = ~closed_image+img
labeled_image, num_features = label(closed_image)

# filling 
#filled_image = morphology.binary_fill_holes(img)
filled_image = morphology.binary_fill_holes(img)#,structure=np.ones((5,5)))
difference_filled = ~filled_image+img
##### CHANGE OF SIGNS DUE TO INVERTED THRESHOLDING!! 

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
opened_image = skimorph.area_opening(img, 100) # area_threshold, manual adjust 
difference = ~opened_image+img
# opening thins the mesh so the inverse expression produces visible output i.e. ~img+opened_image 

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('binary image')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(opened_image, cmap='gray')
plt.title('opened image')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(difference, cmap='gray')
plt.title('difference image')
plt.axis('off')
plt.show();




