from meshure.actimg import get_ActImg
import numpy as np
import matplotlib.pyplot as plt

""" Test theta=0 and theta=30. """

actimg = get_ActImg('testUTR_3miFRIifov1_01.tif', 'actin_meshwork_analysis/meshwork/test_data/')
actimg.normalise()
actimg.steerable_gauss_2order(theta=0,sigma=2,visualise=False)
np.max(actimg.manipulated_stack) # cv2 and scipy 0.18660 vs matlab 0.1865   V
np.min(actimg.manipulated_stack) # cv2 and scipy -0.2817174 vs matlab -0.2818   V
np.mean(actimg.manipulated_stack) # cv2 and scipy -2.58174e-05 vs matlab -2.5817e-05   V

np.sum(actimg.manipulated_stack > 0) # scipy 213852?? cv2 208343 vs matlab 177882
plt.hist(actimg.manipulated_stack.ravel());plt.show();

actimg = get_ActImg('testUTR_3miFRIifov1_01.tif', 'actin_meshwork_analysis/meshwork/test_data/')
actimg.normalise()
actimg.steerable_gauss_2order(theta=30,sigma=2,visualise=False)
np.max(actimg.manipulated_stack) # cv2 .17327 vs matlab 0.1733   V
np.min(actimg.manipulated_stack) # cv2 -0.275751 vs matlab -0.2758   V
np.mean(actimg.manipulated_stack) # cv2 -2.58166e-05 vs matlab -2.5817e-05   V

np.sum(actimg.manipulated_stack > 0) # scipy ? cv2 208659 vs matlab 177046 !!!
plt.hist(actimg.manipulated_stack.ravel());plt.show();


""" Bottom line: do we use non-maximum suppression to thin edges or do we not....
non max supp used by steerablej source (cpp source is a pain to read)
they implement custom non max supp 
my ideas seem to suck
could it be better to simply take max along axis 0 to get strongest response out of all theta responses
"""


actimg = get_ActImg('3min_FOV3_decon.tif', 'actin_meshwork_analysis/process_data/sample_data/CARs')
actimg._visualise_oriented_filters(np.arange(0,360,15),2)

actimg.nuke()
actimg.normalise()
actimg.steerable_gauss_2order_thetas((0,90),2,[3,5],False)
actimg.z_project_min()
actimg.threshold(0.002)
actimg.visualise_stack('manipulated')


actimg.nuke()
actimg.normalise()
ims = actimg.steerable_gauss_2order(theta=0,sigma=2,substack=[3,5],visualise=True,tmp=True)

all_responses = ims['response'].copy()
max_resp = np.max(np.array(all_responses), 0)
mean_resp = np.mean(np.array(all_responses), 0)
diff = mean_resp - max_resp
max_resp = max_resp > 0.002
mean_resp = mean_resp > 0.002
plt.subplot(1,3,1)
plt.imshow(max_resp,cmap='gray')
plt.title('max')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(mean_resp, cmap='gray')
plt.title('mean')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(diff, cmap='gray')
plt.title('diff')
plt.axis('off')
plt.show()


from scipy import ndimage
all_frames = ims['response'].copy()
image = all_frames[0]
# Use the max filter to make a mask
roi = 1
size = 2 * roi + 1
image_max = ndimage.maximum_filter(image, size=size, mode='nearest')
plt.imshow(image_max); plt.show();
mask = (image == image_max)
image *= mask
plt.imshow(image); plt.show(); 


actimg.nuke()
actimg.normalise()
ims = actimg.steerable_gauss_2order(theta=60,sigma=2,substack=[3,5],visualise=True,tmp=True)

all_frames2 = ims['response'].copy()
image2 = all_frames2[0]
# Use the max filter to make a mask
roi = 1
size = 2 * roi + 1
image_max2 = ndimage.maximum_filter(image2, size=size, mode='nearest')
plt.imshow(image_max2); plt.show();
mask2 = (image2 == image_max2)
image2 *= mask2
plt.imshow(image2); plt.show(); 

plt.subplot(1,2,1)
plt.imshow(image_max, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(image_max2, cmap='gray')
plt.show()

max_out = np.max(np.array([image_max,image_max2]),0)
plt.subplot(1,3,1)
plt.imshow(image_max, cmap='gray')
plt.title('theta=0')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(image_max2, cmap='gray')
plt.title('theta=60')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(max_out, cmap='gray')
plt.title('max of two')
plt.axis('off')
plt.show()



# Optionally find peaks above some threshold
peak_threshold = 0.02
image_t = (image > peak_threshold) * 1
plt.imshow(image_t); plt.show(); 
# get coordniates of peaks
f = np.transpose(image_t.nonzero())




actimg.nuke()
actimg.normalise()

thetas=np.arange(15,360,120)
results = dict.fromkeys(thetas)
for angle in thetas:
    results[angle] = actimg.steerable_gauss_2order(theta=angle, substack=[3,5],sigma=2,visualise=False,tmp=True)
response_stack = np.array([value['response'] for key, value in results.items()])

out=[]
for img in np.rollaxis(response_stack,0):
    image = img[0]
    # Use the max filter to make a mask
    roi = 1
    size = 2 * roi + 1
    image_max = ndimage.maximum_filter(image, size=size, mode='nearest')
    out.append(image_max)
    plt.imshow(image_max); plt.show();
    mask = (image == image_max)
    image *= mask
    plt.imshow(image); plt.show(); 

maxim = np.max(np.array(out),0)
plt.imshow(maxim);plt.show();
plt.imshow(maxim > 0.005);plt.show();







import numpy as np
import matplotlib.pyplot as plt 
from tifffile import imread
from scipy import io
from actin_meshwork_analysis.scratch_misc._steerable_gaussian_debugging import steerable_gauss_2order

"""- run steerable filter in matlab on an actin image and see output  
"""

impath = 'meshwork/test_data/testUTR_3miFRIifov1_01.tif'
test_im = imread(impath)

steerable_gauss_2order(test_im, sigma=2, theta=90)

normalised = (test_im-np.min(test_im))/(np.max(test_im)-np.min(test_im))

res = [0]*6
for n, angle in enumerate(np.arange(0,360,60)):
    res[n] = steerable_gauss_2order(normalised, sigma=2, theta=angle, visualise=False, return_stack=True)

mean = np.mean(np.asarray(res),0)
plt.imshow(mean, cmap='gray'); plt.show()

average_norm = 'tests/test_files/average_norm.mat'
average_mat = io.loadmat(average_norm)['J2']
norm_mean = np.mean(average_mat, 2)
plt.imshow(norm_mean, cmap='gray'); plt.show();


# shapes and vals
norm_mean.shape == mean.shape
np.allclose(norm_mean, mean, atol=1e-8)

# plot
minmin = np.min([np.min(norm_mean), np.min(mean)])
maxmax = np.max([np.max(norm_mean), np.max(mean)])
plt.figure(figsize=(10,4))
plt.subplots_adjust(hspace=0.01, vspace=0, top=0)
plt.subplot(1,3,1)
plt.imshow(norm_mean, cmap='gray', vmax=maxmax, vmin=minmin)
plt.title('matlab output')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(mean, cmap='gray', vmax=maxmax, vmin=minmin)
plt.title('python output')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(mean-norm_mean,cmap='gray', vmax=maxmax, vmin=minmin)
plt.title('difference')
plt.axis('off')
plt.suptitle('comparing output for averaging over $\\theta$=[0,60,120,180,240,300]\nnp.allclose(matlab_mean, pyhton_mean, atol=1e-8)==True', fontsize=15)
plt.show();


### raw 90 deg doesnt match

test_im = imread(impath)
matpy = steerable_gauss_2order(test_im, sigma=2, theta=90)


raw_90_2_path = 'tests/test_files/raw_90_2.mat'
raw_90_2 = io.loadmat(raw_90_2_path)['J2']
plt.imshow(raw_90_2, cmap='gray'); plt.show();


# shapes match 
raw_90_2.shape == matpy.shape

np.allclose(raw_90_2, matpy, atol=1e-3)
plt.imshow(matpy-raw_90_2, cmap='gray'); plt.show();
[np.min(raw_90_2), np.min(matpy)]
[np.max(raw_90_2), np.max(matpy)]

minmin = np.min([np.min(raw_90_2), np.min(matpy)])
maxmax = np.max([np.max(raw_90_2), np.max(matpy)])

plt.figure(figsize=(10,4))
plt.subplots_adjust(hspace=0.01, vspace=0, top=0)
plt.subplot(1,3,1)
plt.imshow(raw_90_2, cmap='gray', vmax=maxmax, vmin=minmin)
plt.title('matlab output')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(matpy, cmap='gray', vmax=maxmax, vmin=minmin)
plt.title('python output')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(matpy-raw_90_2,cmap='gray', vmax=maxmax, vmin=minmin)
plt.title('difference')
plt.axis('off')
plt.suptitle('comparing output for averaging over $\\theta$=[0,60,120,180,240,300]\nnp.allclose(matlab_mean, pyhton_mean, atol=1e-8)==True', fontsize=15)
plt.show();



#### NORMALISED FRAMES DO MATCH???? wtf

test_im = imread(impath)
normalised = (test_im-np.min(test_im))/(np.max(test_im)-np.min(test_im))

res = [0]*6

for n, angle in enumerate(np.arange(0,360,60)):
    res[n] = steerable_gauss_2order(normalised, sigma=2, theta=angle)


average_norm = 'tests/test_files/average_norm.mat'
average_mat = io.loadmat(average_norm)['J2']
norm_mean = np.mean(average_mat, 2)


for i in range(6):
    print(np.allclose(norm_mean[:,:,i], res[i], atol=1e-15))
    minmin = np.min([np.min(norm_mean[:,:,i]), np.min(res[i])])
    maxmax = np.max([np.max(norm_mean[:,:,i]), np.max(res[i])])
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(norm_mean[:,:,i], cmap='gray', vmax=maxmax, vmin=minmin)
    plt.title('matlab output')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(res[i], cmap='gray', vmax=maxmax, vmin=minmin)
    plt.title('python output')
    plt.axis('off')
    plt.suptitle('comparing output for averaging over $\\theta$=[0,60,120,180,240,300]', fontsize=15)
    plt.show();


# shapes match 
norm_mean.shape == mean.shape

np.allclose(norm_mean, mean, atol=1e-8)

minmin = np.min([np.min(norm_mean), np.min(mean)])
maxmax = np.max([np.max(norm_mean), np.max(mean)])



# res = response_stack[0], oriented_filter, g0, G2a, G2b, G2c
list(map(np.min, res))
list(map(np.max, res))
# the difference is in the response, oriented filter and Gs are calculated correctly / identically  

# res = response_stack[0], I2a, I2b, I2c
list(map(np.min, res)) # all zeros
list(map(np.max, res)) # all max intensity 65535, 65535, 65535
# vs in matlab in the same order 10959, 11984, 8689, 10959
### WITH CV2 [10959.0, 11984, 8689, 10959]

plt.figure(figsize=(10,4))
for n, (image, title) in enumerate(zip(res, ['I2a', 'I2b', 'I2c'])):
    plt.subplot(1,3,n+1)
    plt.imshow(image, cmap='gray', vmax=65535, vmin=0)
    plt.title(title)
    plt.axis('off')
plt.show();

res[1].dtype 
# output data type for I2s is unit16 whereas response is float64 



raw_path = 'tests/test_files/raw.mat'
norm_path = 'tests/test_files/normalised.mat'
raw_env = io.loadmat(raw_path)
norm_env = io.loadmat(norm_path)
norm_env.keys()

def plot_things_matlab(nplots, keys, titles):
    raw = [raw_env.get(key) for key in keys]
    norm = [norm_env.get(key) for key in keys]
    plt.figure(figsize=(2*nplots,4))
    for n, (image, title) in enumerate(zip([*raw, *norm], titles)):
        plt.subplot(1,nplots,n+1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.colorbar()
    plt.show();

plot_things_matlab(2, 'J', ['raw J', 'norm J'])

plot_things_matlab(6, ['G2a', 'G2b', 'G2c'],
    ['raw G2a', 'raw G2b', 'raw G2c', 'norm G2a', 'norm G2b', 'norm G2c'])

def compare_mats_matlab(keys):
    raw = [raw_env.get(key) for key in keys]
    norm = [norm_env.get(key) for key in keys]
    for n, (raw_v, norm_v) in enumerate(zip(raw, norm)):
        print(keys[n], "   ", np.allclose(raw_v, norm_v))

compare_mats_matlab(['G2a', 'G2b', 'G2c', 'g0', 'oriented_filter'])

compare_mats_matlab(['I', 'I2a', 'I2b', 'I2c'])
