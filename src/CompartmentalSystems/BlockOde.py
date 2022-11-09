from typing import Callable, List, Tuple
import numpy as np
from .PiecewiseBlockIvp import PiecewiseBlockIvp
from .BlockRhs import BlockRhs


class BlockOde:
    """
    Helper class to build a system from functions that operate on blocks of
    the state_variables.
    """

    def __init__(
        self,
        time_str: str,
        block_names_and_shapes: List[Tuple[str, Tuple[int]]],
        functionss: List[List[Tuple[Callable, List[str]]]],
        disc_times: Tuple[int] = ()
    ):
        self.time_str = time_str
        self.block_names_and_shapes = block_names_and_shapes
        self.functionss = functionss
        self.disc_times = disc_times

    def check_block_exists(self, block_name):
        if block_name not in set(self.block_names).union(self.time_str):
            raise Exception("There is no block with this name")

    def blockIvp(
        self,
        start_blocks: List[Tuple[str, np.ndarray]]
    ) -> PiecewiseBlockIvp:
        """
        Extends the system to an initial value problem by adding startvalues
        for the blocks.
        It checks that the names of the names of the blocks coincide.
        """
        
        for ind, tup in enumerate(self.block_names_and_shapes):
            name, shape = tup
            assert(name == start_blocks[ind][0])
            assert(shape == start_blocks[ind][1].shape)

        block_rhss = [
            BlockRhs(
                self.time_str,
                func_tups
            )
            for func_tups in self.functionss
        ]

        return PiecewiseBlockIvp(
            start_blocks,
            block_rhss,
            self.disc_times
        )
