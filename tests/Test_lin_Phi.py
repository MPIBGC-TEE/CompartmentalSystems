
from testinfrastructure.InDirTest import InDirTest
from testinfrastructure.helpers import pe
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  
from CompartmentalSystems.smooth_model_run import SmoothModelRun 

class TestLinPhi(InDirTest):
    # this wants to move to TestSmoothModelRun
    def test_stateTransitionOperator_from_linear_System_by_skew_product_system(self):
        # We first check the result for a linear system, where the whole procedure is not neccesary
        k_0_val=1
        k_1_val=2
        x0_0=np.float(1)
        x0_1=np.float(1)
        delta_t=np.float(1)
        # 
        var(["x_0","x_1","k_0","k_1","t","u"])
        #
        inputs={
             0:u
            ,1:u
        }
        outputs={
             0:x_0*k_0
            ,1:x_1*k_1
        }
        internal_fluxes={}
        srm=SmoothReservoirModel(
                 state_vector       =[x_0,x_1]
                ,time_symbol        =t
                ,input_fluxes       =inputs
                ,output_fluxes      =outputs
                ,internal_fluxes    =internal_fluxes
        )
        # for this system we know the state transition operator to be a simple matrix exponential
        def Phi_ref(times):
            if isinstance(times,np.ndarray): 
                #3d array
                res=np.zeros((nr_pools,nr_pools,len(times)))
                res[0,0,:]=np.exp(-k_0_val*times)
                res[1,1,:]=np.exp(-k_1_val*times)
            else: 
                #2d array for scalar time
                res=np.zeros((nr_pools,nr_pools))
                res[0,0]=np.exp(-k_0_val*times)
                res[1,1]=np.exp(-k_1_val*times)
            return(res)

        t_max=4
        times = np.linspace(0, t_max, 11)
        parameter_dict = {
             k_0: k_0_val
            ,k_1: k_1_val
            ,u:1

        }
        func_dict={}
        #B=srm.compartmental_matrix
        nr_pools=srm.nr_pools
        nq=nr_pools*nr_pools
        #tup=(t,)+tuple(srm.state_vector)
        #B_num=numerical_function_from_expression(B,tup,parameter_dict,{})
        sol_rhs=numerical_rhs2(
             srm.state_vector
            ,srm.time_symbol
            ,srm.F
            ,parameter_dict
            ,func_dict
        )
        B_sym=srm.compartmental_matrix
        #
        tup=(t,)+tuple(srm.state_vector)
        B_func=numerical_function_from_expression(B_sym,tup,parameter_dict,func_dict)
        x_i_start=0
        x_i_end=nr_pools
        Phi_1d_i_start=x_i_end
        Phi_1d_i_end=(nr_pools+1)*nr_pools
        int_i_start=Phi_1d_i_end
        int_i_end=int_i_start+nr_pools
        #
        def Phi_rhs(t,x,Phi_1d):
            B=B_func(t,*x)
            Phi_cols=[Phi_1d[i*nr_pools:(i+1)*nr_pools] for i in range(nr_pools)]
            Phi_ress=[np.matmul(B,pc) for pc in Phi_cols]
            return np.stack([np.matmul(B,pc) for pc in Phi_cols]).flatten()
        #
        #
        start_x= np.array([x0_0,x0_1])
        start_Phi_1d=np.identity(nr_pools).reshape(nr_pools**2)
        start_Int_Phi_u=np.zeros(nr_pools)
        t_span=(0,t_max)
        #        
        # even more compactly the same system
        block_ivp=BlockIvp(
            time_str='t'
            ,start_blocks  = [('sol',start_x),('Phi',start_Phi_1d)]
            ,functions = [
                 (sol_rhs,['t','sol'])
                ,(Phi_rhs,['t','sol','Phi'])
             ]
        )
        s_block_ivp=block_ivp.solve(t_span=t_span)
        t_block_rhs    = block_ivp.get_values("t"         ,t_span=t_span)
        sol_block_rhs  = block_ivp.get_values("sol"       ,t_span=t_span)
        Phi_block_rhs  = block_ivp.get_values("Phi"       ,t_span=t_span)
        Phi_block_rhs_mat   =Phi_block_rhs.reshape(nr_pools,nr_pools,len(t_block_rhs))
        #print(block_ivp.get("sol",t_span=t_span))
        # for comparison solve the original system 
        sol=solve_ivp(fun=sol_rhs,t_span=(0,t_max),y0=start_x,max_step=delta_t,method='LSODA')
        # and make sure that is identical with the block rhs by using the interpolation function
        # for the block system and apply it to the grid that the solver chose for sol
        sol_func  = block_ivp.get_function("sol"       ,t_span=t_span,dense_output=True)
        self.assertTrue(np.allclose(sol.y,sol_func(sol.t),rtol=1e-2))
        # check Phi
        self.assertTrue(np.allclose(Phi_block_rhs_mat,Phi_ref(t_block_rhs),atol=1e-2))

        fig=plt.figure(figsize=(7,7))
        ax1=fig.add_subplot(2,1,1)
        ax1.plot(sol.t,               sol.y[0,:],'o',color='red' ,label="sol[0]")
        ax1.plot(sol.t,               sol.y[1,:],'x',color='red' ,label="sol[1]")
        #
        ax1.plot(sol.t,               sol_func(sol.t)[0,:],'*',color='blue' ,label="sol[0]")
        ax1.plot(sol.t,               sol_func(sol.t)[1,:],'+',color='blue' ,label="sol[1]")
        #
        ax1.plot(t_block_rhs, sol_block_rhs[0,:],'*',color='green',label="sol_block_rhs[0]")
        ax1.plot(t_block_rhs, sol_block_rhs[1,:],'+',color='green',label="sol_block_rhs[1]")
        #
        ax1.legend()

        ax2=fig.add_subplot(2,1,2)
        ax2.plot(t_block_rhs  ,Phi_ref(t_block_rhs)[0,0,:],'o',color='red'  ,label="Phi_ref[0,0]")
        ax2.plot(t_block_rhs  ,Phi_ref(t_block_rhs)[1,1,:],'x',color='red'  ,label="Phi_ref[1,1]")

        ax2.plot(t_block_rhs, Phi_block_rhs_mat[0,0,:],'*',color='green',label="Phi_block_rhs_mat[0,0]")
        ax2.plot(t_block_rhs, Phi_block_rhs_mat[1,1,:],'+',color='green',label="Phi_block_rhs_mat[1,1]")
        ax2.legend()
        fig.savefig("solutions.pdf")
        
