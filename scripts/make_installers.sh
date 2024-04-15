#!/usr/bin/env python3
from string import Template
from pathlib import Path
import sys
import os
dir_path=Path(__file__).parent
file_name=Path(os.path.basename(__file__))
sys.path.insert(0,dir_path)
import shortesRelativePath as srp
from difflib import Differ

out_path=Path(".")
srp=srp.rp(s=out_path,t=dir_path)

t1=Template("# This file has been automatically  procuded by ${fn} , do not change manually, but rather ${fn}")
t2=Template("""#If we use conda or one of its derivatives we install as
# many of the dependencies via conda and leave only the src packages for pip
# (for which there are no conda packages}.
# This leaves conda in control of your environment and avoides confusion.
# If you do not use conda but only pip, you do not preinstall any requirements since pip will also find and install them from the setup.py file directly.
${command} install -y -c conda-forge --file requirements.test --file requirements.doc --file requirements.non_src pip 
""")
t3=Template(
"""# We install the dependencies that are not on pypy directly from github repos
# This is not possible with conda (therefore we use pip here)
# Do not do this (comment the following lines) 
# if you have checked out these repos and installed the code in developer mode 
${command} install --upgrade pip
${command} install git+https://github.com/MPIBGC-TEE/testinfrastructure.git#egg=testinfrastructure
${command} install git+https://github.com/MPIBGC-TEE/LAPM.git#egg=LAPM

# The following line installs the package without checking out the repository directly from github
# if you want to install it in develop mode (after checking out the repo) comment this line and
# use "$command -e ." instead in the same directory where this file lives. 
${command} install git+https://github.com/MPIBGC-TEE/CompartmentalSystems.git#egg=CompartmentalSystems
""")
txt1 = t1.substitute(
    fn=srp.joinpath(file_name)
)
def write(command,suffix,txt):
    script_file_name=f"install_{command}.{suffix}"
    with Path(script_file_name).open("w") as f:
        f.write(txt)

for suffix in ["sh","bat"]:
    txt3 = t3.substitute(
        command="pip" if suffix =="sh" else f"call pip",
    )
    for command in ["conda","mamba","micromamba", "pip"]:

        txt2 = t2.substitute(
            command=command if suffix =="sh" else f"call {command}",
        )
        
        write(command, suffix, txt1 + txt2 + txt3)
    for command in ["pip"]:
        write(command, suffix, txt1 + txt3)
