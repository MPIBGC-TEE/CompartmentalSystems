from typing import Callable, List, Tuple
import numpy as np

from scipy.integrate import solve_ivp
# from .myOdeResult import solve_ivp_pwc
from .BlockRhs import BlockRhs

class BlockIvp:
    """
    Helper class to build initial value systems from functions that operate
    on blocks of the state_variables.
    """
    def __init__(
        self,
        start_blocks: List[Tuple[str, np.ndarray]],
        block_rhs: BlockRhs
    ):
        self.array_dict = {tup[0]: tup[1] for tup in start_blocks}
        names = [sb[0] for sb in start_blocks]
        start_arrays = [sb[1] for sb in start_blocks]

        sizes = [a.size for a in start_arrays]
        nb = len(sizes)
        r = range(nb)
        indices = [0] + [sum(sizes[:(i+1)]) for i in r]
        self.index_dict = {names[i]: (indices[i], indices[i+1]) for i in r}
        block_shapes = [(n, a.shape) for (n, a) in start_blocks] 
        self.rhs = block_rhs.flat_rhs(block_shapes)
        self.start_vec = np.concatenate([a.flatten() for a in start_arrays])


    def block_solve(self, t_span, first_step=None, **kwargs):
        sol_obj = solve_ivp(
            fun=self.rhs,
            t_span=t_span,
            y0=self.start_vec,
            **kwargs
        )

        def block_sol(block_name):
            start_array = self.array_dict[block_name]
            lower, upper = self.index_dict[block_name]
            time_dim_size = sol_obj.y.shape[-1]
            tmp = sol_obj.y[lower:upper, :].reshape(
                start_array.shape+(time_dim_size,)
            )
            # solve_ivp returns an array that has time as the LAST dimension
            # but our code usually expects it as FIRST dimension
            # Therefore we move the last axis to the first position
            return np.moveaxis(tmp, -1, 0)

        block_names = self.index_dict.keys()
        block_sols = {block_name: block_sol(block_name)
                      for block_name in block_names}
        return block_sols

    def block_solve_functions(self, t_span, first_step=None, **kwargs):
        kwargs['dense_output'] = True
        sol_obj = solve_ivp(
            fun=self.rhs,
            t_span=t_span,
            y0=self.start_vec,
            **kwargs
        )

        def block_sol(block_name):
            start_array = self.array_dict[block_name]
            lower, upper = self.index_dict[block_name]

            def func(times):
                tmp = sol_obj.sol(times)[lower:upper]
                if isinstance(times, np.ndarray):
                    res = tmp.reshape(
                        (start_array.shape+(len(times),))
                    )
                    return np.moveaxis(res, -1, 0)
                else:
                    return tmp.reshape(start_array.shape)

            # solve_ivp returns an array that has time as the LAST dimension
            # but our code usually expects it as FIRST dimension
            # Therefore we move the last axis to the first position
            return func

        block_names = self.index_dict.keys()
        block_sols = {block_name: block_sol(block_name)
                      for block_name in block_names}
        return block_sols
