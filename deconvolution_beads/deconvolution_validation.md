# Validating measured PSF (for **calibration with beads**)

The fluorescent beads (size = 40 nm) were imaged using the same setup as actual imaging.  

These can be used to mimic SF 1 in @Fritzsche2017. 

----- 

Calculating PSF to confirm whether the values from the measured PSF are significantly different from the Huygens software defaults. If they differ by a large margin, the beads will have to be reimaged.  

Problems:  
- not entirely clear what a "bead" is in the images. best guess is something three pixels wide. 

## Method: fitting a 2D Gaussian

- either using Picasso (python-based software) or programmatically in Python 

## TODO

- [ ] SF1 from paper 
    - [ ] shows the maximum resolution of STED setup


- [ ] find all beads images
- [ ] fit 2D gaussian onto beads (we've selected 4-5 "beads"); get average of those values 
- [ ] can use picasso or fit directly in python 
    - [ ] use lineprofile either in python or in imagej 
- [ ] send sigmas to Sabrina, Ceci, and Olivia

----

# Resolution 

$$\Delta r = \frac{\lambda}{2 NA} \frac{1} {\sqrt{1 + \frac{I^{max}_{STED}}{I_{sat}}}}$$ 

where:  
- $NA$ = numerical aperture of the objective lens 
- $I$ = STED laser power 
- $I_{sat}$ = fluorophore saturation intensity 
- the second factor determines how small the point spread function will be as a function of the saturation factor (which scales with STED intensity)

- finding (distilling) PSF = inverse problem (much like deconvolution)
- need to average multiple beads due to photon noise
- need to correct for bead size 
- PSF distiller measures FWHM 
- graph of average FWHM over saturation factors --> overlay of measured and theoretical PSF
    - inverse square root function 
- one field of view is sufficient  


## Theoretical PSF 

- estimated from microscope parameters 
- saturation factor = OBF/LIF metadata
    - dye properties, depletion laser type and intensity 
- STED depletion wavelength - OBF/LIF metadata 
- STED immunity fraction (3-10% will fluoresce through the doughnut depletion beam = confocal-like cloud around PSF) - OBF metadata 
- STED 3X - power used for depletion in axial direction 

## Drift correction

- thermal drift around 10 nm must be accounted for in STED imaging
- 


## Save deconvolved data as HDF5 or ICS 