import pytest, os
from pytest import raises
import numpy as np
from meshure.actimg import ActImg

""" TODO
    - No negative tests!!
    - test_data_copies_at_steerable_gauss_2order
    - test visualisation functions
    - fix threshold test - currently threshold is inverse (as per previous implementation)
"""

"""
Examples
--------
def test_satmap_invalid_data(meta_15, fov_15, centre_15, shape_15):
    with pytest.raises(TypeError) as err:
        SatMap("", meta_15, fov_15, centre_15, shape_15)
    assert "Input data must be an array" in str(err.value)
"""

def test_data_copies_at_instantiation(anActImg, random_image): 
    assert anActImg.depth == 3
    assert anActImg.shape == (10,30)
    assert anActImg.title == 'test.tiff'
    assert np.isclose(anActImg.image_stack[0], random_image[0]).all()
    
def test_data_copies_at_normalise(z_proj_im, normalise_res):
    actimg = ActImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20)
    actimg.normalise()
    assert (actimg.image_stack[0] == z_proj_im[0]).all()
    assert (actimg.manipulated_stack[0] == normalise_res).all()


def test_data_copies_at_z_proj_min(z_proj_im, min_proj_res):
    actimg = ActImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20)
    actimg.normalise()
    actimg.z_project_min()
    assert (actimg.image_stack[0] == z_proj_im[0]).all()
    assert (actimg.manipulated_stack == min_proj_res).all()


def test_data_copies_at_z_proj_max(z_proj_im, max_proj_res):
    actimg = ActImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20)
    actimg.normalise()
    actimg.z_project_max()
    assert (actimg.image_stack[0] == z_proj_im[0]).all()
    assert (actimg.manipulated_stack == max_proj_res).all()


def test_data_copies_at_threshold(z_proj_im, threshold_max_proj_res):
    actimg = ActImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20)
    actimg.normalise()
    actimg.z_project_max()
    actimg.threshold(0.6)
    assert (actimg.image_stack[0] == z_proj_im[0]).all()
    assert (actimg.manipulated_stack == threshold_max_proj_res).all()


def test_min_proj_substack(z_proj_substack, min_proj_res, min_grad_res):
    actimg = ActImg(z_proj_substack, 'test.tiff', (10,10), 3, False, 20) 
    actimg.normalise()
    actimg.z_project_min([1,2])
    # should match test_data_copies_at_z_proj_min
    assert (actimg.manipulated_stack==min_proj_res).all()
    actimg2 = ActImg(z_proj_substack, 'test.tiff', (10,10), 3, False, 20) 
    actimg2.normalise()
    actimg2.z_project_min([2,3])
    # should match gradient fixture
    assert np.allclose(actimg2.manipulated_stack, min_grad_res, rtol=1e-4)


def test_max_proj_substack(z_proj_substack, max_proj_res, max_grad_res):
    actimg = ActImg(z_proj_substack, 'test.tiff', (10,10), 3, False, 20) 
    actimg.normalise()
    actimg.z_project_max([1,2])
    # should match test_data_copies_at_z_proj_min
    assert (actimg.manipulated_stack==max_proj_res).all()
    actimg2 = ActImg(z_proj_substack, 'test.tiff', (10,10), 3, False, 20) 
    actimg2.normalise()
    actimg2.z_project_max([2,3])
    # should match gradient fixture
    assert np.allclose(actimg2.manipulated_stack, max_grad_res, rtol=1e-4)


# def test_data_copies_at_steerable_gauss_2order(z_proj_im):
#     pass


""" Test visualisation functions. """
# imtype="",ind=1,save=False,dest_dir="",

@pytest.mark.parametrize('args, exp_error', [
    ([0,1,False], "imtype must be a string"),
    (['some','some',False], "ind must be an integer"),
    (['some',1,'some'], "save must be a boolean")
    ])
def test_visualise_type_errors(z_proj_im, args, exp_error):
    actimg = ActImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20)
    with pytest.raises(TypeError) as err:
        actimg.visualise(*args)
    assert exp_error in str(err.value)
    

@pytest.mark.parametrize('args, exp_error', [
    (['some',1,False], "not recognised; imtype must be one of ['original', 'manipulated']"),
    (['original',0,False], "ind must be an integer in range (1, 2)"),
    (['original',4,False], "ind must be an integer in range (1, 2)"), 
    (['original',1,False,'notadir'], "Directory not recognised:")
    ])
def test_visualise_value_errors(z_proj_im, args, exp_error):
    actimg = ActImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20)
    with pytest.raises(ValueError) as err:
        actimg.visualise(*args)
    assert exp_error in str(err.value)


#imtype=s, substack=ss, save=False, dest_dir=""
@pytest.mark.parametrize('args, exp_error', [
    ([0,[1,2],False], "imtype must be a string"),
    (['original',[1,2],'some'], "save must be a boolean")
    ])
def test_visualise_stack_type_errors(z_proj_im, args, exp_error):
    actimg = ActImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20)
    with pytest.raises(TypeError) as err:
        actimg.visualise_stack(*args)
    assert exp_error in str(err.value)
    

@pytest.mark.parametrize('args, exp_error', [
    (['some',[1,2],False], "not recognised; imtype must be one of ['original', 'manipulated']"),
    (['original',[1,2,3],False], "substack has to be a list of length=2, specifying a range"),
    (['original',[0,1],False], "substack must be a list of integers in range (1, 2)"),
    (['original',[1,10],False], "substack must be a list of integers in range (1, 2)"),
    (['original',[1,2],False,'notadir'], "Directory not recognised:")
    ])
def test_visualise_stack_value_errors(z_proj_im, args, exp_error):
    actimg = ActImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20)
    with pytest.raises(ValueError) as err:
        actimg.visualise_stack(*args)
    assert exp_error in str(err.value)


""" Test helper methods. """

def test_nuke(z_proj_im):
    actimg = ActImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20) 
    actimg.normalise()
    actimg.z_project_max()
    assert str(actimg) != str(ActImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20))
    actimg.nuke()
    assert str(actimg) == str(ActImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20))


def test_call_hist(z_proj_im):
    actimg = ActImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20) 
    actimg.normalise()
    actimg.z_project_min()
    assert actimg._history == ['normalise', 'z_project_min']
    actimg.nuke()
    actimg.normalise() 
    actimg.z_project_max()
    assert actimg._history == ['normalise', 'z_project_max']



""" Test steerable filter: 

test_steer_tmp(z_proj_im):
    actimg = ActImg(z_proj_im, 'test.tiff', (10,10), 2, False, 20) 
    actimg.normalise()
    res = actimg.steerable_gauss_2order([1,2],visualise=False,tmp=True)
    assert actimg._history == ['normalise']
    actimg.steerable_gauss_2order([1,2],visualise=False,tmp=False)
    assert actimg._history == ['normalise', 'steerable_gauss_2order']
    
"""