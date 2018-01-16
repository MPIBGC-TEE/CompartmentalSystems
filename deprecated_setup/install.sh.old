# Although the following packages are mentioned in setup.py as dependencies
# the install in the order determined by setuptools seems to fail sometimes, probably
# due to unstated dependencies of numpy and matlotlib.
# This script helps in such cases but is not intended to replace setup.py
# 

pip3 install --upgrade pip

pip3 install concurrencytest
pip3 install jupyter
pip3 install jupyter_contrib_nbextensions
pip3 install jupyter_nbextensions_configurator
pip3 install matplotlib
pip3 install numpy
pip3 install plotly
pip3 install scipy
pip3 install sympy
pip3 install tqdm

pip3 install git+https://github.com/MPIBGC-TEE/LAPM
pip3 install git+https://github.com/mamueller/testinfrastructure

python3 setup.py develop

# enable jupyter notebook nbextensions
jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user
jupyter nbextension enable python-markdown/main

