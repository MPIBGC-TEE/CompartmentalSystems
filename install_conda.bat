# This file has been automatically procuded by the function: 
# testinfrastructure.make_installers.make_installers_cmd 
# and those of the following files that are present in the root
# folder of the repo: 
# requirements.non_src, 
# requierements.conda_extra, 
# requirements.test, 
# requirements.doc, 
# requirements.github, 
# pkg.github 
#
# If you have testinfrastructure installed
# you can change the requirement files and recreate all the install 
# scripts automatically by running the command: 
# $make_installers 


# If we use conda or one of its derivatives we
# - install as many of the dependencies via conda 
# - block pip from installing them by the "--no-deps" flag
#  -and leave only the src packages for pip (for which there are no conda packages}.
# This leaves conda in control of your environment and prevents pip from
# reinstalling packages that are already installed by conda but not
# detected by pip. 
# This happens only for some packages (e.g.  # python-igraph) 
# but is a known problem at the time of writing and does affect us.
call  conda install -y -c conda-forge matplotlib mpmath numpy plotly scipy sympy tqdm frozendict networkx pandas python-igraph pycairo pip


# To run the tests (which is recommended but not necessary for the package to work) 
# you need some extra packages which can be installed by uncommenting the following line:
call  conda install -y -c conda-forge dask[distributed] jupyterlab jupytext nbformat nbmake notebook pytest zarr


# If you want to develop the package including the documentation you need some extra tools which can be installed by uncommenting the following line:
# call  conda install -y -c conda-forge sphinx sphinx-autodoc-typehints


# We install the dependencies that are not on pypy directly from github repos
# This is not possible with conda (therefore we use pip here)
# Do not do this (comment the following lines) 
# if you have checked out these repos and installed the code in developer mode
call  pip install --no-deps git+https://github.com/MPIBGC-TEE/testinfrastructure.git#egg=testinfrastructure git+https://github.com/MPIBGC-TEE/LAPM.git#egg=LAPM

# The following line installs the package without checking out the repository directly from github
call  pip install --no-deps git+https://github.com/MPIBGC-TEE/CompartmentalSystems.git@test#egg=CompartmentalSystems

# if you want to install it in develop mode (after checking out the repo) comment the previous line and
# execute the following line instead, in the same directory where this file lives. 
# call  pip install --no-deps -e .
