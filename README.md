# actinmeshure: Quantify cytoskeletal actin remodelling

<!-- ----
badges to be included 
---- -->

AchinMeshure is a small library of functions which aim to quantify actin remodelling from super-resolution STED images of cytoskeletal actin. 

The library includes classes and helpers to: 

- read in images and extract metadata into `ActinImg` instances
- manipulate, visualise, or save the images 
- run batch analyses on entire directories with data using the `ActinImgCollection` class
- summarise and visualise batch processing results, written into HTML files (requires Pandoc)


## Installation from source 

To install ActinMeshure, download the code from the repository, navigate to the destination and run:  

`$ pip install .`

If you wish to edit/develop the package, run it directly from source using:

`$ pip install -e .  # Run once to add package to Python path`

## License

Copyright 2023, Seda Radoykova.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Citation 

If you find this project useful to your work, please, cite it as: 

> Radoykova, S., 2023. ActinMeshure: Quantify cytoskeletal actin remodelling. https://github.com/sedaradoykova/actin_mesh_measures

