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

Installation simply via the install script `install.sh`.
Use either `develop` or `install` as additional parameter, it will be 
passed to the `python3 setup.py` call.

Be sure to have [LAPM](https://github.com/MPIBPG-TEE/LAPM) installed.
Further required packages can be found in the install script.

---

Jupyter notebook examples
-------------------------

- [PNAS Supporting Information](http://htmlpreview.github.io/?https://github.com/MPIBGC-TEE/CompartmentalSystems/blob/master/notebooks/PNAS/notebook.html)
- [Analysis of a nonlinear global carbon cycle model (html)](http://htmlpreview.github.io/?https://github.com/MPIBGC-TEE/CompartmentalSystems/blob/master/notebooks/nonl_gcm_3p/nonl_gcm_3p.html)
- [Analysis of a nonlinear global carbon cycle model (ipynb)](notebooks/nonl_gcm_3p/nonl_gcm_3p.ipynb)


