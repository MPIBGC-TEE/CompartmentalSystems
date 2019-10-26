from testinfrastructure.InDirTest import InDirTest
#from testinfrastructure.helpers import pe
import numpy as np

from CompartmentalSystems.BlockIvp import BlockIvp

class TestBlockIvp(InDirTest):
    def test_build_rhs(self):
        b_s=BlockIvp.build_rhs(
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
#--------------------------------------------------------------------------
#
    def test_block_rhs_versus_block_ivp(self):
        pass
        #s_block_rhs=solve_ivp(
        #    fun=block_rhs(
        #         time_str='t'
        #         ,X_blocks  = [('sol',nr_pools),('Phi',nr_pools*nr_pools),('Int_Phi_u',nr_pools)]
        #         ,functions = [
        #             (sol_rhs,['t','sol'])
        #             ,(Phi_rhs,['t','sol','Phi'])
        #             ,(Int_phi_u_rhs,['t','Phi'])
        #          ]
        #    )
        #    ,t_span=t_span
        #    ,y0=np.concatenate([ start_x,start_Phi_1d,start_Int_Phi_u])
        #)
        #t_block_rhs         =s_block_rhs.t
        #sol_block_rhs       =s_block_rhs.y[x_i_start:x_i_end,:]
        #Phi_block_rhs       =s_block_rhs.y[Phi_1d_i_start:Phi_1d_i_end,:]
        #Phi_block_rhs_mat   =Phi_block_rhs.reshape(nr_pools,nr_pools,len(t_block_rhs))
        #int_block_rhs       =s_block_rhs.y[int_i_start:int_i_end,:]
        #
        ## even more compactly the same system
        #block_ivp=BlockIvp(
        #    time_str='t'
        #    ,start_blocks  = [('sol',start_x),('Phi',start_Phi_1d),('Int_Phi_u',start_Int_Phi_u)]
        #    ,functions = [
        #         (sol_rhs,['t','sol'])
        #        ,(Phi_rhs,['t','sol','Phi'])
        #        ,(Int_phi_u_rhs,['t','Phi'])
        #     ]
        #)
        ## but we can also acces single blocks of the result
        #self.assertTrue(np.array_equal( t_block_rhs     ,x_phi_ivp.get("t"         ,t_span=t_span)))
        #self.assertTrue(np.array_equal( sol_block_rhs   ,x_phi_ivp.get("sol"       ,t_span=t_span)))
        #self.assertTrue(np.array_equal( Phi_block_rhs   ,x_phi_ivp.get("Phi"       ,t_span=t_span)))
        #self.assertTrue(np.array_equal( int_block_rhs   ,x_phi_ivp.get("Int_Phi_u" ,t_span=t_span)))
        ## we can get the same solution object we get from solve_ivp
        ##print(x_phi_ivp.get("sol",t_span=t_span))
        ##
        ##
        


