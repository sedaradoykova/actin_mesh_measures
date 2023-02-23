# TODO 

-------

## Actin mesh size/density analysis 

- [X] What is the **cytosolic** actin network? Is it 0-1 $\mu$ m? 
    - cytosolic and basal network 
    - > we defined it individually for each cell 
    - [pseudocolor cv](https://plantcv.readthedocs.io/en/stable/visualize_pseudocolor/)
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
- [ ] adapt morphological operations to the data (lol)

- [x] make a new class which can help perform the analysis 
    - [X] guide parametrisation and enable automatic html generation
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

- correlation is what i thought convolution was (going top-bottom and left-right)
- [ ] make intermediate Gaussian figures to be used in write up
- [ ] write up bits of the methods section ... [not urgent]
- [X] take filter in six orientations and average of outputs 
- [X] output of cv2 and scipy is consistent for normalised data - the averages match
    - but it is not consistent with the intermediates of the matlab code (see _scratch_gaussian_debugging.py)
- [X] confirm that theta value is not the problem --> compare matlab to python theta conversion 
    - V: theta conversion is identical 
- [X] test with theta=0 (and theta=30) 
    - check mins and max of images:
    - V: the image histograms are very very similar for theta=0 and 30 
    - V: min, max, mean = match; fraction > 0 is close but not identical.... 
- [X] look at steerableJ source code  
- [X] WIP: steerable filter post-processing: 
    - [Jacob and Unser (2004)](https://ieeexplore.ieee.org/document/1307008) use non-maximum suppression as a post-processing step. it is a procedure which thins edges 
        - non max supp used by steerablej source (cpp source is a pain to read, took an entire day)
        - they implement custom non max supp, i don't understand how they do it... C++ 
    - they also conclude that higher order derivatives of the gaussian detect fewer artifacts (see the BIG demo, re_gauss.pmd)
        - **do we need a means of detecting noises and false detections? compare to a baseline?**
    - Bottom line: do we use non-maximum suppression to thin edges or do we not....
    - compare mean to max of thetas responses !!!!! 
        - could it be better to simply take max along axis 0 to get strongest response out of all theta responses? 
        - max produces a thicker mesh, mean produces thinner.... (which makes sense) 
    - [Miruna's paper, data analysis docx](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001152#s4) ([github](https://github.com/alexcarisey/ActinMeshAnalyzer))use various post-processing steps in an interactive image processing pipeline in matlab
        - great idea of using line profile to threshold
        - the software has some bugs and doesn't work with our images very well 
    - current ideas summarised in lab_files/progress_Feb.ppt

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
        - performed analysis without thresholding


#### TODO following meeting 

- [X] check (relative) number of CARs 
    | Cell type     | Count | 
    | ---           | --- | 
    | Untransduced    | 22 |
    | CAR_dual        | 25 |
    | CAR_antiCD19    | 11 |
    | CAR_antiCD22     | 7 |
    | CAR_UNTR         | 6 |
- [X] fix max projection to be basal and cytosolic planes respectively 
- [X] are six orientations too many?? .... check with three thetas 
    - the output is similar, if not identical
    - it appears that two orthogonal orientations e.g. 0 and 90 deg are sufficient to cover and reproduce pattern - the rest is repeated periodically
    -  steerable source uses angles between 0-180
- [-] focus on thresholding dynamically, forget comparison to matlab
- [-] take 3 images, get meshwork size/density 
- [X] check miruna's paper before trying something in python
    - note: the plugin does more things than we need 
    - [actin mesh analyser](https://github.com/alexcarisey/ActinMeshAnalyzer)
    - **see above**
- [X] upload and email all images/prelim results
    - there is a problem with the frame names and displayed images - noticed in CAR - lagging one image behind?? 
        - fixed when re-run analysis 


## 22/02 meeting notes and new TODO

- [ ] get C++ to work on windows 
    - tried installing on my machine, reqs: visual studio 2022, vcpkg, GSL (GNU scientific library), C++ compiler and extensions for VSCode
    - problem: installation cannot find gsl/gsl_some_func.h file even though it is recognised by VSC in the script
    - same problem for matlab: -I/src/gsl/include not recognised as a command 
    - 
- [ ] max projection: 
    - normalise img and whack up contrast (as in SF2 A/D)
    - z=0 (starting of basal plane) to z=1 micrometer (check z-step size)
        - include frames n to n+m (m being the frame at depth = 1mu)
    - 
- [ ] dynamic thresholding: get line profiles from different positions and aggregate 
    - try lines crossing ROI **OR** lines which are in the periphery **OR** boxes in the periphery
        - or background boxes: discard line if order of magnitude difference in intensity values? another cell's noise... 
    - fit Gaussian to aggregated line profile and take mean + 1/2 st devs
        - Signal must be Gaussian :) 
    - ideal background = no cell parts, then noise = Poisson distributed 
- [ ] steerable filter post-processing 
    - invert steerable response average **OR** use an inverse filter with values `< threshold` kept (instead of `> threshold`)
        - OR use mean - st dev 
    - for inversion of response: normalise [0,1] and invert 