--- 
There is a handful of pdfs in the ./reading dir. 
--- 

# Steerable filter introduction

- [Steerable Pyramid: Theory of Steerable Filters](https://rafat.github.io/sites/wavebook/advanced/steer.html)

### Second-order steerable filters

- 2-nd derivative Gaussian filters of different orientations
- examples illustrated = oriented spatial functions 
- filters are convolved with the images and data derived from along the axons is retained 
    - non-maximum suppression removes data which does not lie on ridges of high intensity ~ to surrounding landscape 

### Interpolation 

- [information is lost any time an image is rotated](https://www.cambridgeincolour.com/tutorials/image-interpolation.htm)
- hence, interpolation functions

### Separability and SVD
- [SVD basis of steerable filter separability in X-Y, Bart Wronski](https://bartwronski.com/2020/02/03/separate-your-filters-svd-and-low-rank-approximation-of-image-filters/)

# Code 

- [andreydung/Steerable-filter](https://github.com/andreydung/Steerable-filter/blob/master/perceptual/filterbank.py)
- [DSF-CNN paper github](https://github.com/simongraham/dsf-cnn)
- [Matlab, Steerable filter (steerable_filter), Pang2022](https://uk.mathworks.com/matlabcentral/fileexchange/44956-steerable-filter)
- [Matlab, Steerable Gaussian Filters (Steerable Filters), Lanman2022](https://uk.mathworks.com/matlabcentral/fileexchange/9645-steerable-gaussian-filters)


## Papers 

- [Steerable Filters and Local Analysis of Image Structure](https://apps.dtic.mil/sti/pdfs/ADA605046.pdf)
- [See Fig. 1C](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3579723/)
    > "Images were imported into MATLAB (MathWorks, Natick, MA) and convolved with a bank of three 2D 2nd-derivative Gaussian filters of width 1.8 μm. The maximum filter response was calculated. The method is described in full in Freeman and Adelson [11]."

## Initial reading around it 

- [scipy gaussian filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html)
- [EPFL BIG steerable](http://bigwww.epfl.ch/demo/steerable/download.html)
- [steerableJ](https://biii.eu/steerablej)
    - [steerable (not J)](image.pngimage.pnghttp://bigwww.epfl.ch/demo/steerable/)
    - [source code: C++, Python, Matlab](https://github.com/francois-a/steerable/blob/master/src/steerableDetector.h)
    - [ridge detection demo](http://bigwww.epfl.ch/demo/steerable/ridge.html)
- [building recognition project, implemented steerable filter](https://github.com/mitchdull/sfbr)
- [imageJ process manual](https://imagej.nih.gov/ij/docs/guide/146-29.html)
- [3D imagej plugin source](https://github.com/pam66/steerable3D/blob/master/src/main/java/eu/marbilab/imagej/Steerable3D_.java)


## CNNs and stuff

- [Convolutional Neural Network: Feature Map and Filter Visualization](https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c)
- [Harmonic Networks: Deep Translation and Rotation Equivariance](https://arxiv.org/abs/1612.04642)
- [Learning Steerable Filters for Rotation Equivariant CNNs](https://openaccess.thecvf.com/content_cvpr_2018/papers/Weiler_Learning_Steerable_Filters_CVPR_2018_paper.pdf)



### Chapters 

- [Chapter 5 - Microaneurysms Detection for Early Diagnosis of Diabetic Retinopathy Using Shape and Steerable Gaussian Features](https://www.sciencedirect.com/science/article/pii/B9780128169483000052)
- [Chapter 4 - Design of Integrated Road Perception and Lane Detection System for Driver Intention Inference](https://www.sciencedirect.com/science/article/pii/B978012819113200004X)
- [Quantitative Analysis of Nuclear Lamins Imaged by Super-Resolution Light Microscopy](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6524165/)
    - > The steerableDetector function was applied to reconstructed SIM images and a rastered STORM image with a fourth order filter. The Euclidean minimum spanning tree, Delaunay triangulation, and Voronoi tessellation of SMLM STORM data were computed using the Computational Geometry functions and the Bioinformatics Toolbox of MATLAB after loading the molecule list from Nikon Elements. T
- [Texture classification filters (group, Oxford)](https://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html)

## Other

- [Scalespace or Differential Geometry](https://www.vanosta.be/scalespace.htm)

## Local contrast enhancement 

- used in original steerable filter enhancement 
- [local contrast enhancement](https://www.cambridgeincolour.com/tutorials/local-contrast-enhancement.htm#:~:text=Local%20contrast%20enhancement%20works%20similarly,mapped%20when%20sharpening%20an%20image.)

