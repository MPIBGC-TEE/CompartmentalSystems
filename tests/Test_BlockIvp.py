from testinfrastructure.InDirTest import InDirTest
from testinfrastructure.helpers import pe
import numpy as np

from CompartmentalSystems.BlockIvp import BlockIvp
from CompartmentalSystems.helpers_reservoir import block_rhs

class TestBlockIvp(InDirTest):
    def test_block_rhs(self):
        b_s=block_rhs(
             time_str='t'
            ,X_blocks=[('X1',5),('X2',2)]
            ,functions=[
                 ((lambda x   : x*2 ),  ['X1']    )
                ,((lambda t,x : t*x ),  ['t' ,'X2'])
             ])   
        # it should take time and 
        example_X=np.append(np.ones(5),np.ones(2))
        res=b_s(0,example_X)
        ref=np.array([2, 2, 2, 2, 2, 0, 0])
        self.assertTrue(np.array_equal(res,ref))
        # solve the resulting system
