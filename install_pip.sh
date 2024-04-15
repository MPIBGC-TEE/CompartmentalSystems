# This file has been automatically  procuded by scripts/make_installers.sh , do not change manually, but rather scripts/make_installers.sh# We install the dependencies that are not on pypy directly from github repos
# This is not possible with conda (therefore we use pip here)
# Do not do this (comment the following lines) 
# if you have checked out these repos and installed the code in developer mode 
pip install git+https://github.com/MPIBGC-TEE/testinfrastructure.git#egg=testinfrastructure
pip install git+https://github.com/MPIBGC-TEE/LAPM.git#egg=LAPM

# The following line installs the package without checking out the repository directly from github
# if you want to install it in develop mode (after checking out the repo) comment this line and
# use "pip -e ." instead in the same directory where this file lives. 
pip install git+https://github.com/MPIBGC-TEE/CompartmentalSystems.git#egg=CompartmentalSystems
