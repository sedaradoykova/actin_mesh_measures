import numpy as np
import matplotlib.pyplot as plt 
from tifffile import imread
from scipy import io
from meshwork.steerable_gaussian_debugging import steerable_gauss_2order



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
