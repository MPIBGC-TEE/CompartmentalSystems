"""Module for symbolical treatment of smooth 14C reservoir models.

This module handles the symbolic treatment of compartmental/reservoir/pool
models.
It does not deal with numerical computations and model simulations,
but rather defines the underlying structure of the respective model.

All fluxes or matrix entries are supposed to be SymPy expressions.
*Smooth* means that no ``Piecewise`` or ``DiracDelta`` functions should be
involved in the model description.

Counting of compartment/pool/reservoir numbers starts at zero and the
total number of pools is :math:`d`.
"""

from sympy import Matrix, eye
from copy import copy

from .smooth_reservoir_model import SmoothReservoirModel


class Error(Exception):
    """Generic error occurring in this module."""
    pass


class SmoothReservoirModel_14C(SmoothReservoirModel):
    """General class of smooth 14C reservoir models.

    Attributes:
        decay_symbol (SymPy symbol): The model's decay symbol.
    """

    def __init__(self, srm, decay_symbol, Fa):
        """Construct and return a :class:`SmoothReservoirModel_14C` instance that
           models the 14C component of the original model.

        Args:
            srm (SmoothReservoirModel): The original model.
            decay_symbol (SymPy symbol): The symbol of the 14C decay rate.
            Fa (SymPy Function): The atmospheric C14 fraction function.
        """
        B_14C = copy(srm.compartmental_matrix) - decay_symbol*eye(srm.nr_pools)
        u = srm.external_inputs
        u_14C = Matrix(srm.nr_pools, 1, [expr*Fa for expr in u])

        srm_14C = super().from_B_u(
            srm.state_vector,
            srm.time_symbol,
            B_14C,
            u_14C
        )

        super().__init__(
            srm_14C.state_vector,
            srm_14C.time_symbol,
            srm_14C.input_fluxes,
            srm_14C.output_fluxes,
            srm_14C.internal_fluxes
        )
        self.decay_symbol = decay_symbol

    @property
    def output_fluxes_corrected_for_decay(self):
        d = dict()
        for k, val in self.output_fluxes.items():
            d[k] = val - self.decay_symbol * self.state_vector[k]

        return d
