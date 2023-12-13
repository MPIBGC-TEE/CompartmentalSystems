# automatically created by  ../../scripts/create_install_scripts.py
# from setup.py , requirements.conda.extra and  requirements.src
    mamba install -y sphinx-autodoc-typehints tqdm numpy plotly sympy frozendict scipy python-igraph mpmath matplotlib pip networkx
    pip show testinfrastructure;ret=$? 
if [ $ret -eq 0 ]; then
    echo "allredy installed"
else 
    pip install -e git+https://github.com/MPIBGC-TEE/testinfrastructure.git#egg=testinfrastructure
fi 
pip show lapm;ret=$? 
if [ $ret -eq 0 ]; then
    echo "allredy installed"
else 
    pip install -e git+https://github.com/MPIBGC-TEE/LAPM.git#egg=lapm
fi 
    pip install -e .  