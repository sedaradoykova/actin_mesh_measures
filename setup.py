from setuptools import setup, find_packages

setup(
    name = 'ActinMeshure',
    version = '0.1.0',
    author = 'Seda Radoykova',
    author_email = 'seda.radoykova.19@ucl.ac.uk',
    description = 'This package aims to quantify cytoskeletal actin remodelling from super resolution STED images of T cells.',
    url = 'https://github.com/sedaradoykova/actin_mesh_measures',
    download_url ='https://github.com/sedaradoykova/actin_mesh_measures',
    license = '(C) Simoncelli Lab 2023',
    # packages = find_packages(
    #     include = ['meshwork']  # alternatively: `include=['additional*']`
    #     ),  
    # package_dir = {"": "actin_meshwork_analysis"},
    packages = ['meshure'],
    install_requires = [
        'Cython==0.29.32',
        'h5py==3.8.0',
        'matplotlib==3.5.2',
        'matplotlib_scalebar==0.8.1',
        'numpy==1.21.5',
        'opencv_python==4.6.0.66',
        'pandas==1.4.4', 
        'Pillow==9.4.0',
        'pytest==7.1.2', 
        'PyYAML==6.0',
        'scikit_image==0.19.2',
        'scipy==1.9.1',
        'setuptools==63.4.1',
        'tifffile==2021.7.2',
        'tqdm==4.64.1',
        'myst_parser==1.0.0',
        'sphinx==6.2.0',
        'furo==2023.5.20',
        'nbsphinx==0.9.2',
        'Jinja2==3.0.3',
        'Flask==1.1.1',
        'numpydoc==1.5.0',
        'ipykernel==6.19.2'
    ],
    entry_points = {
        'console_scripts': [
            'cli_name = module_dir.modulepy:function' 
        ]}
)


# https://setuptools.pypa.io/en/latest/userguide/entry_point.html gui and others 