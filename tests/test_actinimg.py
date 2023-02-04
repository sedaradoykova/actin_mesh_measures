import pytest, os
from pytest import raises
import numpy as np
from meshwork.actinimg import ActinImg

""" TODO
    - No negative tests!!
    - test_data_copies_at_steerable_gauss_2order
    - test visualisation functions
"""

"""
Examples
--------
def test_satmap_invalid_data(meta_15, fov_15, centre_15, shape_15):
    with pytest.raises(TypeError) as err:
        SatMap("", meta_15, fov_15, centre_15, shape_15)
    assert "Input data must be an array" in str(err.value)
"""

def test_data_copies_at_instantiation(anActinImg, random_image): 
    assert anActinImg.depth == 3
    assert anActinImg.shape == (10,30)
    assert anActinImg.title == 'test.tiff'
    assert np.isclose(anActinImg.image_stack[0], random_image[0]).all()
    
def test_data_copies_at_normalise(z_proj_im, normalise_res):
    actimg = ActinImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20)
    actimg.normalise()
    assert (actimg.image_stack[0] == z_proj_im[0]).all()
    assert (actimg.manipulated_stack[0] == normalise_res).all()


def test_data_copies_at_z_proj_min(z_proj_im, min_proj_res):
    actimg = ActinImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20)
    actimg.normalise()
    actimg.z_project_min()
    assert (actimg.image_stack[0] == z_proj_im[0]).all()
    assert (actimg.manipulated_stack == min_proj_res).all()


def test_data_copies_at_z_proj_max(z_proj_im, max_proj_res):
    actimg = ActinImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20)
    actimg.normalise()
    actimg.z_project_max()
    assert (actimg.image_stack[0] == z_proj_im[0]).all()
    assert (actimg.manipulated_stack == max_proj_res).all()


def test_data_copies_at_threshold(z_proj_im, threshold_max_proj_res):
    actimg = ActinImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20)
    actimg.normalise()
    actimg.z_project_max()
    actimg.threshold(0.6)
    assert (actimg.image_stack[0] == z_proj_im[0]).all()
    assert (actimg.manipulated_stack == threshold_max_proj_res).all()


def test_min_proj_substack(z_proj_substack, min_proj_res, min_grad_res):
    actimg = ActinImg(z_proj_substack, 'test.tiff', (10,10), 3, False, 20) 
    actimg.normalise()
    actimg.z_project_min([1,2])
    # should match test_data_copies_at_z_proj_min
    assert (actimg.manipulated_stack==min_proj_res).all()
    actimg2 = ActinImg(z_proj_substack, 'test.tiff', (10,10), 3, False, 20) 
    actimg2.normalise()
    actimg2.z_project_min([2,3])
    # should match gradient fixture
    assert np.allclose(actimg2.manipulated_stack, min_grad_res, rtol=1e-4)


def test_max_proj_substack(z_proj_substack, max_proj_res, max_grad_res):
    actimg = ActinImg(z_proj_substack, 'test.tiff', (10,10), 3, False, 20) 
    actimg.normalise()
    actimg.z_project_max([1,2])
    # should match test_data_copies_at_z_proj_min
    assert (actimg.manipulated_stack==max_proj_res).all()
    actimg2 = ActinImg(z_proj_substack, 'test.tiff', (10,10), 3, False, 20) 
    actimg2.normalise()
    actimg2.z_project_max([2,3])
    # should match gradient fixture
    assert np.allclose(actimg2.manipulated_stack, max_grad_res, rtol=1e-4)


# def test_data_copies_at_steerable_gauss_2order(z_proj_im):
#     pass


def test_nuke(z_proj_im):
    actimg = ActinImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20) 
    actimg.normalise()
    actimg.z_project_max()
    assert str(actimg) != str(ActinImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20))
    actimg.nuke()
    assert str(actimg) == str(ActinImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20))


def test_call_hist(z_proj_im):
    actimg = ActinImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20) 
    actimg.normalise()
    actimg.z_project_min()
    assert actimg._history == ['normalise', 'z_project_min']
    actimg.nuke()
    actimg.normalise() 
    actimg.z_project_max()
    assert actimg._history == ['normalise', 'z_project_max']