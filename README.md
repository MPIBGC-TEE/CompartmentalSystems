This is a Python package for compartmental systems.

Python package to deal with compartmental models of the form

$$
\frac{d}{dt}\,x(t) = A(x(t),t)\,x(t) + u(t).
$$

Since most computations are based on the state transition operator $\Phi$ that solves

$$
\frac{d}{dt}\,\Phi(t,s) = A(t)\,\Phi(t,s),\quad \Phi(s,s) = \bf{I},
$$

nonlinear models need to be linearized in the first step. Then the package provides numerical computation of

* age

    * compartmental age densities
    * system age densities
    * compartmental age mean and higher order moments
    * system age mean and higher order moments
    * compartmental age quantiles
    * system age quantiles

* transit time

    * forward and backward transit time densities
    * backward transit time mean and higher order moments
    * forward and backward transit time quantiles

---

[Documentation](http://compartmentalsystems.readthedocs.io/en/latest/)

---

Installation simply via the install script `install.sh`.

---

Jupyter notebook examples
-------------------------

- [Nonlinear global carbon cycle model (html)](notebooks/nonl_gcm_3p.html)
- [Nonlinear global carbon cycle model (ipynb)](notebooks/nonl_gcm_3p.ipynb)



