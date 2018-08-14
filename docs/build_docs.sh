# make sure that our version of sphinx is installed
# pip install -r ~/LAPM/requirements.txt

#execute here
clear;rm -r _autosummary / _build ; sphinx-build -b html . _build
# alternatively (automatically overwrites _build subdir)
# make html 
