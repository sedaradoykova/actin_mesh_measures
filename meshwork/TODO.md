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
    