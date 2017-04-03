Compartmental Systems
=====================

`CompartmentalSystems <https://github.com/MPIBGC-TEE/CompartmentalSystems>`_ is a 
Python package to deal with compartmental models of the form

.. math:: \frac{d}{dt}\,x(t) = B(x(t),t)\,x(t) + u(t).

Since most computations are based on the state transition operator :math:`\Phi` that solves

.. math:: \frac{d}{dt}\,\Phi(t,s) = B(t)\,\Phi(t,s),\quad \Phi(s,s) = \bf{I},

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


Table of Contents
-----------------

.. autosummary::
    :toctree: _autosummary

    ~CompartmentalSystems.smooth_reservoir_model
    ~CompartmentalSystems.smooth_model_run


Jupyter notebook examples
-------------------------

* `Analysis of a nonlinear global carbon cycle model (html) <_downloads/nonl_gcm_3p.html>`_ :download:`. <../notebooks/nonl_gcm_3p/nonl_gcm_3p.html>`
* :download:`Analysis of a nonlinear global carbon cycle model (ipynb) <../notebooks/nonl_gcm_3p/nonl_gcm_3p.ipynb>`


Important Note
--------------

:math:`B(t)=(b_{ij}(t))` is supposed to be a *compartmental matrix* for all times :math:`t`:

* :math:`b_{ii}(t)\leq0` for all :math:`i`
* :math:`b_{ij}(t)\geq0` for :math:`i\neq j`
* :math:`\sum\limits_{i=1}^d b_{ij}(t)\leq 0` for all :math:`j`


----------------------------------

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

