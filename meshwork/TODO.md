# TODO 

-------

## Actin mesh size/density analysis 

- [X] What is the **cytosolic** actin network? Is it 0-1 $\mu$ m? 
    - cytosolic and basal network 
    > we defined it individually for each cell 
- [ ] Use the new deconvolved images to finish actin meshwork analysis.
    - [ ] get mesh size
    - [X] start with untransduced cells; then CARs
    - [-] start with 3/8 min, "pretty" ones 
        - ignore images with B7!! 
- [ ] Untransduced 3/8 min: 
    - [ ] paper Fig 1C = 3 min?
    - [X] check if images are with patterns/ring in that fig 
- [ ] Fig 1A 3D z-projection

----

## Current analysis pipeline priorities

- [X] **Normalise** --> **steer 2o Gauss** --> **min z proj** --> **threshold** 
    - do for all deconvolved images of interest 
- [X] use spreadsheet to choose focal plane for **min z proj** 
- [ ] adapt morphological operations to suit the data (lol)

- [x] make a new class which can help perform the analysis 
    - [ ] guide parametrisation and enable automatic html generation
    - [ ] could be integrated as a command line interface too   

----

## Thresholding steps 

- [X] Is thresholding introducing an artifact? 
    - try running the code on one example cell (until the thresholding) **without applying the steerable filter** 
    - compare == wither steerable filter is introducing too much “artifact” to the image analysis 

- [x] What is an appropriate value? 
    - automatically set ~ image 
    - calculate mean + std of the background signal (ie where there are no cells) 
    - threshold = mean +1.5 (or 2.5x) std
    - [ ] test for one image, calculate mean and std without cell by masking out cell 
        - NaNs --> take mean of mask (normalise-->min_z-->threshold)
    - the value might be similar across cells 
    - [X] done with threshold set to mean BUT NO SD 

---- 

## Steerable filter matters 

- correlation is what i thought covolution was (going top-bottom and left-right)
- [ ] make intermediate Gaussian figures to be used in write up
- [ ] write up bits of the methods section 
- [X] take filter in six orientations and average of outputs 
- [X] output of cv2 and scipy is consistent for normalised data - the averages match
    - but it is not consistent with the intermediates of the matlab code (see _scratch_gaussian_debugging.py)
- [ ] confirm that theta value is not the problem --> compare matlab to python theta conversion 
- [ ] test with theta=0 
    - check mins and max of images 
- [ ] look at steerableJ source code 
- [ ] are six orientations too many?? .... 

----

### 13/03 meeting outline

- **steerable gaussian filter** 
    - i finally understood what the filter is and what it does 
        - visualise intermediates, write tests to check they are consistent 
    - tests to compare matlab and python output (ppt and gaussian_debugging / test_*)
    - cv2 and scipy report consistent results for normalised data 
        - have not tested for raw but there seems to be no point in doing so
- **full pipeline implemented to be automated as a separate class which can process all files in a root_dir**
    - show summary, interactive aspects, parametrisation 
        - pipeline customisation lacks 
    - show results for untransduced, car with focal planes (full analysis)
    - show sample for: 
        - typical pipeline
        - threshold = mean background 
        - typical without steerable gaussian filter 
- **binary threshold** 
    - interactive slider for threshold fine-tuning 
        - exposes that at very low thresholds noise creates a mesh-like structure 
    - is thresholding producing artifacts? 
        - performed analysis without 


- check number of cars (relative)
- check with three thetas 
- forget comparison to matlab 
- focus on thresholding dynamically 
- get meshwork size/density 
- take 3 images 
- check miruna's paper before trying something in python 
    - note: the plugin does more things than we need 
- upload and email all images/prelim results
- fix max projection to be basal and cytosolic planes respectively 