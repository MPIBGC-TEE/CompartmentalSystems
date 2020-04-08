[![Travis-CI Build Status](https://travis-ci.org/MPIBGC-TEE/CompartmentalSystems.svg?branch=master)](https://travis-ci.org/MPIBGC-TEE/CompartmentalSystems)
# CompartmentalSystems: A Python3 package for the analysis of compartmental systems

These systems can be both nonlinear and nonautonomous. Consequently, CompartmentalSystems can be seen
as a companion package of [LAPM](https://github.com/MPIBGC-TEE/LAPM) which deals
with linear autonomous models.
While [LAPM](https://github.com/MPIBGC-TEE/LAPM) also allows explicit symbolic compuations of age distributions 
in compartmental systems, this package is mostly concerned with numerical
computations of

* Age

    * compartmental age densities
    * system age densities
    * compartmental age mean and higher order moments
    * system age mean and higher order moments
    * compartmental age quantiles
    * system age quantiles

* Transit time

    * forward and backward transit time densities
    * backward transit time mean and higher order moments
    * forward and backward transit time quantiles

---

[Documentation](http://compartmentalsystems.readthedocs.io/en/latest/)

---
[Installation]
---
The way you install the package depends on your usage scenario.
- In a new (most likely virtual) environment "./install.sh" will install the package with pinned dependencies  and some useful extras to use it (like jupyter). If this succeeds you can use the package afterwards.
- If you just want to install the package with the versions of some dependencies fixed but do not want it to interfere with your own setup of tools (jupyter) just type 
 " pip3 install -requirements.txt -e . " (This is also one step done by the install.sh script)
- If you want to develop the package or integrate it into your own library and have the required libraries already installed you can use
 "python3 setup.py develop". This is the least invasive scenario since "setup.py" does not contain package versions (on purpose). This way you can test the package with your own set of library versions (e.g. with a more recent sympy). 
It will **not** make sure that the versions are compatible with the package (on purpose).


Be sure to have [LAPM](https://github.com/MPIBGC-TEE/LAPM) installed.
Further required packages can be found in the install script.

---

Jupyter notebook examples
-------------------------

- [PNAS Supporting Information](http://htmlpreview.github.io/?https://github.com/MPIBGC-TEE/CompartmentalSystems/blob/master/notebooks/PNAS/PNAS_notebook.html)
- [Analysis of a nonlinear global carbon cycle model (html)](http://htmlpreview.github.io/?https://github.com/MPIBGC-TEE/CompartmentalSystems/blob/master/notebooks/nonl_gcm_3p/nonl_gcm_3p.html)
- [Analysis of a nonlinear global carbon cycle model (ipynb)](notebooks/nonl_gcm_3p/nonl_gcm_3p.ipynb)

## Information for developers

### Notebooks
We have several objectives, with partially contradictory requirements that necessiate a considerate approach. We want 
1.  the notebooks to be under version control,
1.  to make sure that the code in the notebooks exectutes without error and that
1.  the results of the notebooks do not change unnoticed due to changes in the code called from the notebook.
   
Version controlling a .ipnyb file is difficult for the following reasons:
* the file gets messed up accidentally because the autosaving adds misleading information about the most 
  recent changes and leads to overwriting intended changes by another contributor.
* the data (e.g. plots) and metadata contained in the notebook are noisy and hide the intended changes.

To avoid the misleading information about which change is most recent we turn of autosaving. (The following seems to work for jupyter notebook as well as jupyterlab)
1. To find the Jupyter configuration folder execute the following cod3:
```python
from jupyter_core.paths import jupyter_config_dir
jupyter_config_dir()  # ~/.jupyter on my machine
```
2. create sub-folder custom, and create file custom.js within it:
   i.e. '~/.jupyter/custom/custom.js'
   Put the following line in custom.js:
```javascript
IPython.notebook.set_autosave_interval(0);
```
Save file and restart the Jupyter Notebook server (main app).When opening a notebook you should see "Autosave disabled" briefly appearing in the right side of the menu bar.

To avoid the binary output in the ipynb file  we use the [jupytext](https://github.com/mwouts/jupytext) package, which allows us to accompany every notebook file with a ligth python version which is stripped of the metadata and output and only contains the inputs (code) to reproduce the output
This is the file that we put under version control. 
The jupytext extension allows us to load this file directly into jupyter as a notebook. It automatically creates the ipynb file.( by using the```*.py``` file for the input and - if present in your working copy - an older version of the ```*.ipynb``` file for the output, in which case you should run it to make the inputs and outputs consistent)
In general we do **not** put the ```ipynb``` file under verion control.

The testing of the notebooks can be done wiht ```pytest``` if also the ```nbval``` package is present  
```bash
pytest --nbval PNAS_notebook_jupytext.ipynb
```
If the `*.ipynb` file is not (yet) present we can create it from the `*.py` version created by jupytext
```bash
jupytext --to notebook PNAS_notebook_jupytext.py 
```
This process can be automated for the notebooks under development.

However, since the s`*.py` cript (intentionally) does not contain data, it can
not be used to detect changes in the output of a notebook that might accidentally occure if the libraries used by the  notebook change. The `pytest` framework claims to be able to detect those changes, but produced false positives for 
one plot in the pnas notebook, which furthermore did not run at all if it had not been run before on the same machine.
If the notebook is freshly produced from the `*.py` file everything seems to work and the `pytest` can be relied upon to 
check if the code is still running without errors. 
On Antakya the test needs 24s.

