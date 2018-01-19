# CompartmentalSystems: A Python3 package for the analysis of compartmental systems

These systems can be both nonlinear and nonautonomous. Consequently, this package can be seen
as an extension of [LAPM](https://github.com/MPIBGC-TEE/LAPM) which deals
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
- In a new (most likely virtual) environment "./install.sh" will install the package with pinned dependencies  and some usefull extras to use it (like jupyter). If this succeeds you can use the package afterwards.
- If you just want to install the package with the versions of some dependencies fixed but do not want it to interfere with your own setup of tools (jupyter) just type 
 " pip3 install -rrequirements.txt -e . " (This is also one step done by the install.sh script )
- If you want to develop the package or integrate it into your own library and have the required libraries already installed you can use
 "python3 setup.py develop". This is the least invasive scenario since "setup.py" does not contain package versions (on purpose). This way you can test the package with your own set of library versions (e.g. with a more recent sympy). 
It will **not** make sure that the versions are compatible with the package (on purpose).


Be sure to have [LAPM](https://github.com/MPIBPG-TEE/LAPM) installed.
Further required packages can be found in the install script.

---

Jupyter notebook examples
-------------------------

- [PNAS Supporting Information](http://htmlpreview.github.io/?https://github.com/MPIBGC-TEE/CompartmentalSystems/blob/master/notebooks/PNAS/PNAS_notebook.html)
- [Analysis of a nonlinear global carbon cycle model (html)](http://htmlpreview.github.io/?https://github.com/MPIBGC-TEE/CompartmentalSystems/blob/master/notebooks/nonl_gcm_3p/nonl_gcm_3p.html)
- [Analysis of a nonlinear global carbon cycle model (ipynb)](notebooks/nonl_gcm_3p/nonl_gcm_3p.ipynb)


