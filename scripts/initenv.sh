# This script wants to be sourced!
# source initenv.sh (otherwise you will not see the benefit of the set paths)

# This script destroys and recreates a virtual environment in the users home dir
# It assumes that python-3.6.4 has been successfully 
# installed under $HOME/opt along with the tcl and tk libs

myOpt=~/opt
tclDir=$myOpt/tcl8.6.8
export PATH=$tclDir/bin:$PATH
export TCL_LIBRARY=$tclDir/lib/tcl8.6
export LD_LIBRARY_PATH=${tclDir}/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=${tclDir}/lib:$LD_LIBRARY_PATH

tkDir=$myOpt/tk8.6.8
export PATH=$tkDir/bin:$PATH
export TKPATH=$tkDir/bin
export TK_LIBRARY=$tkDir/lib/tk8.6
export LD_LIBRARY_PATH=${tkDir}/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=${tkDir}/lib:$LD_LIBRARY_PATH

python3Dir=$myOpt/Python-3.6.4
export PATH=$python3Dir/bin:$PATH
export LD_LIBRARY_PATH=${python3Dir}:$LD_LIBRARY_PATH
export LIBRARY_PATH=${python3Dir}:$LD_LIBRARY_PATH

myPython3Venv="${HOME}/env-3.6.4-opt"
rm -rf $myPython3Venv
mkdir -p $myPython3Venv
python3.6 -m venv $myPython3Venv
source $myPython3Venv/bin/activate
