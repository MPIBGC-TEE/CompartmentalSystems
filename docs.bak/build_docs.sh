# make sure that our version of sphinx is installed
# pip install -r ~/LAPM/requirements.txt
# the topmost file is index.rst   

# to check that the documentation is correctly built by readthedocs
# it is usefull to have an account there (at least if something went wrong)
# There are some

#execute here
clear;rm -r _autosummary / _build ; sphinx-build -b html . _build
# alternatively (automatically overwrites _build subdir)
# make html 
