# This file has been automatically procuded by the function: 
# testinfrastructure.make_installers.make_installers_cmd  
# We install the dependencies that are not on pypy directly from github repos
# This is not possible with conda (therefore we use pip here)
# Do not do this (comment the following lines) 
# if you have checked out these repos and installed the code in developer mode 
 pip install --upgrade pip
 pip install  -r requirements.github

# The following line installs the package without checking out the repository directly from github
# if you want to install it in develop mode (after checking out the repo) comment this line and
# use " pip -e ." instead in the same directory where this file lives. 
 pip install  -r pkg.github
 # If you do not use conda but only pip, you do not have to
preinstall 
# any requirements since pip will also find and install them from the
# setup.py file directly.  So we only install stuff explicitly that
# is not a dependency of the package but necessary for testing.
 pip install -r requirements.test
