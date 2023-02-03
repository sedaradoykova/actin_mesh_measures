from meshwork.steerable_gaussian_debugging import steerable_gauss_2order
import pytest
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from scipy import io

""" Tests for results and internal intermediates for a single theta value.
        Matlab results saved from workspace after running notes.m
 """

@pytest.fixture 
def test_im():
    impath = 'tests/test_files/testUTR_3miFRIifov1_01.tif'
    print('returning image')
    return imread(impath)

@pytest.fixture 
def test_im_norm(test_im):
    return (test_im-np.min(test_im))/(np.max(test_im)-np.min(test_im))

@pytest.fixture
def raw_matlab_res():
    return io.loadmat('tests/test_files/s2_th90_raw.mat')


@pytest.fixture
def norm_matlab_res():
    return io.loadmat('tests/test_files/s2_th90_normalised.mat')

@pytest.fixture
def raw_python_res(test_im):
    return steerable_gauss_2order(test_im, sigma=2, theta=90, visualise=False, return_dict=True)

@pytest.fixture
def norm_python_res(test_im_norm):
    return steerable_gauss_2order(test_im_norm, sigma=2, theta=90, visualise=False, return_dict=True)


# def plot_things_matlab(keys, titles, norm_matlab_res, raw_matlab_res):
#     nplots = len(keys)*2
#     raw = [raw_matlab_res.get(key) for key in keys]
#     norm = [norm_matlab_res.get(key) for key in keys]
#     plt.figure(figsize=(2*nplots,4))
#     for n, (image, title) in enumerate(zip([*raw, *norm], titles)):
#         plt.subplot(1,nplots,n+1)
#         plt.imshow(image, cmap='gray')
#         plt.title(title)
#         plt.axis('off')
#         plt.colorbar()
#     plt.savefig()

# plot_things_matlab('J', ['raw J', 'norm J'])
# plot_things_matlab(['G2a', 'G2b', 'G2c'],
#     ['raw G2a', 'raw G2b', 'raw G2c', 'norm G2a', 'norm G2b', 'norm G2c'])


@pytest.mark.parametrize("keys,expect",[
    (['G2a', 'G2b', 'G2c', 'g0', 'oriented_filter'], True), 
    (['I', 'I2a', 'I2b', 'I2c', 'J'], False)],
)
def test_compare_mats_matlab(norm_matlab_res, raw_matlab_res, keys, expect):
    raw = [raw_matlab_res.get(key) for key in keys]
    norm = [norm_matlab_res.get(key) for key in keys]
    for n, (raw_v, norm_v) in enumerate(zip(raw, norm)):
        assert np.allclose(raw_v, norm_v) is expect


@pytest.mark.parametrize(
    "key",['G2a', 'G2b', 'G2c', 'g0', 'oriented_filter', 'I', 'I2a', 'I2b', 'I2c', 'J']
)
def test_compare_raw_matpy(raw_matlab_res, raw_python_res, key):
    mat, pyth = raw_matlab_res.get(key), raw_python_res.get(key)
    assert np.allclose(mat, pyth, atol=1e-5)
    assert mat.dtype == pyth.dtype


@pytest.mark.parametrize(
    "key",['G2a', 'G2b', 'G2c', 'g0', 'oriented_filter', 'I', 'I2a', 'I2b', 'I2c', 'J']
)
def test_compare_norm_matpy(norm_matlab_res, norm_python_res, key):
    mat, pyth = norm_matlab_res.get(key), norm_python_res.get(key)
    assert np.allclose(mat, pyth, atol=1e-5)
    assert mat.dtype == pyth.dtype



""" Tests comparing the mean output of thetas [0,60,120,180,240,300].
        Matlab results obtained from running 
"""

@pytest.fixture
def raw_matlab_res_x6():
    return io.loadmat('tests/test_files/s2_th0_300_6_raw.mat')['J2']

@pytest.fixture
def raw_matlab_res_x6_mean(raw_matlab_res_x6):
    return np.mean(raw_matlab_res_x6, 2)

@pytest.fixture
def norm_matlab_res_x6():
    return io.loadmat('tests/test_files/s2_th0_300_6_normalised.mat')['J2']

@pytest.fixture
def norm_matlab_res_x6_mean(norm_matlab_res_x6):
    return np.mean(norm_matlab_res_x6, 2)



@pytest.fixture
def raw_python_res_x6(test_im):
    res = [0]*6
    for n, angle in enumerate(np.arange(0,360,60)):
        res[n] = steerable_gauss_2order(test_im, sigma=2, theta=angle, visualise=False, return_stack=True)
    return res

@pytest.fixture
def raw_python_res_x6_mean(raw_python_res_x6):
    return np.mean(np.asarray(raw_python_res_x6),0)


@pytest.fixture
def norm_python_res_x6(test_im_norm):
    res = [0]*6
    for n, angle in enumerate(np.arange(0,360,60)):
        res[n] = steerable_gauss_2order(test_im_norm, sigma=2, theta=angle, visualise=False, return_stack=True)
    return res

@pytest.fixture
def norm_python_res_x6_mean(norm_python_res_x6):
    return np.mean(np.asarray(norm_python_res_x6),0)



def test_compare_matpy_raw_x6_mean(raw_matlab_res_x6_mean, raw_python_res_x6_mean):
    assert raw_matlab_res_x6_mean.shape == raw_python_res_x6_mean.shape
    assert np.allclose(raw_matlab_res_x6_mean, raw_python_res_x6_mean, atol=1e-3, rtol=1-5)

def test_compare_matpy_norm_x6_mean(norm_matlab_res_x6_mean, norm_python_res_x6_mean):
    assert norm_matlab_res_x6_mean.shape == norm_python_res_x6_mean.shape
    assert np.allclose(norm_matlab_res_x6_mean, norm_python_res_x6_mean, atol=1e-3, rtol=1-5)