# This file has been automatically  procuded by scripts/make_installers.sh , do not change manually, but rather scripts/make_installers.sh#If we use conda or one of its derivatives we install as
# many of the dependencies via conda and leave only the src packages for pip
# (for which there are no conda packages}.
# This leaves conda in control of your environment and avoides confusion.
# If you do not use conda but only pip, you do not preinstall any requirements since pip will also find and install them from the setup.py file directly.
micromamba install -y -c conda-forge --file requirements.test --file requirements.doc --file requirements.non_src --file requirements.conda_extra
# We install the dependencies that are not on pypy directly from github repos
# This is not possible with conda (therefore we use pip here)
# Do not do this (comment the following lines) 
# if you have checked out these repos and installed the code in developer mode 
pip install --upgrade pip
pip install git+https://github.com/MPIBGC-TEE/testinfrastructure.git#egg=testinfrastructure
pip install git+https://github.com/MPIBGC-TEE/LAPM.git#egg=LAPM

# The following line installs the package without checking out the repository directly from github
# if you want to install it in develop mode (after checking out the repo) comment this line and
# use "pip -e ." instead in the same directory where this file lives. 
pip install git+https://github.com/MPIBGC-TEE/CompartmentalSystems.git#egg=CompartmentalSystems
