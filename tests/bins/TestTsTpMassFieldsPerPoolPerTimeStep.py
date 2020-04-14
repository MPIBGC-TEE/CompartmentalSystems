# vim: set ff=unix expandtab ts=4 sw=4:
import unittest
from testinfrastructure.InDirTest import InDirTest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad 
from scipy.interpolate import interp1d
from sympy import var,sin,cos,sympify

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.pwc_model_run import PWCModelRun

from CompartmentalSystems.bins.density_algorithm import losses,gains
from CompartmentalSystems.bins.TsTpMassField import TsTpMassField 
from CompartmentalSystems.bins.TsTpMassFields import TsTpMassFieldsPerPool,TsTpMassFieldsPerPipe
from CompartmentalSystems.bins.CompatibleTsTpMassFieldsPerPool import CompatibleTsTpMassFieldsPerPool
from CompartmentalSystems.bins.TsTpDeathRateField import TsTpDeathRateField
from CompartmentalSystems.bins.TsTpMassFieldsPerPoolPerTimeStep import TsTpMassFieldsPerPoolPerTimeStep
from CompartmentalSystems.bins.getsizeof import total_size


class TestTsTpMassFieldsPerPoolPerTimeStep(InDirTest):

#fixme:
    #only the plot methods special to this collection should be added here

##########################################################################
##########################################################################
##########################################################################
    @unittest.skip("obsolete. Better test the iterator")
    def test_mean_age_distribution_for_BW(self):
        # create the model
        var("t, k_01,k_10,k_0o,k_1o")
        var("C_0,C_1")
        state_variables=[C_0,C_1] # order is important
        inputs={
            #0:sin(t)+2,#input to pool 0
            #1:cos(t)+2 #input to pool 1
            0:sympify(2),#input to pool 0
            1:sympify(0) #input to pool 1
            }
        outputs={
            0:k_0o*C_0**3,#output from pool 0
            1:k_1o*C_1**3 #output from pool 0
            }
        internal_fluxes={
            (0,1):k_01*C_0*C_1**2, #flux from pool0  to pool 1
            (1,0):k_10*C_0*C_1 #flux from pool1  to pool 0
            }
        time_symbol=t
        mod=SmoothReservoirModel(state_variables,time_symbol,inputs,outputs,internal_fluxes)

        #set the time step size
        tss=.1
        #create a Model run
        self.params={
            k_01:1/100,
            k_10:1/100,
            k_0o:1/2,
            k_1o:1/2
        }
        
        start_values=[1,2]
        times=np.arange(100)*tss   # time grid forward
        mr=PWCModelRun(mod,self.params,start_values,times)

        
        # now create initial age distributions 
        # since we start with system age 0 we start with very
        # small fields indeed
        # pool 0
        x,y=1,1
        s=(x,y)
        age_dist_0=TsTpMassField(np.zeros(s),tss)
        age_dist_0[0,0]=start_values[0]
        
        # pool 1
        x1,y1=1,1
        s=(x1,y1)
        age_dist_1=TsTpMassField(np.zeros(s),tss)
        age_dist_1[0,0]=start_values[1]

        # initialize the combination (this would adjust for different system ages)
        initial_plains=CompatibleTsTpMassFieldsPerPool([age_dist_0,age_dist_1])
        # we now build the deathrate functions
        # note that the factories depend
        # on the solution funtions 

        # produce the output deathrate functions
        def external_death_rate_maker(sender,func,solfs):
            def wrapper(field,t):
                tss=field.tss
                loss=quad(func,t,t+tss)[0]
                stock=solfs[sender](t)
                relative_loss=loss/stock
                #print("stock:=",stock)
                #print("loss:=",loss)
                #print("ext_relative_loss:=",relative_loss)
                dr=TsTpDeathRateField(relative_loss*np.ones(field.shape),tss)
                return(dr)
            return(wrapper)

        external_death_rate_functions=dict()
        solfs=mr.sol_funcs()
        for sender,func in mr.output_flux_funcs().items():
            external_death_rate_functions[sender]=external_death_rate_maker(sender,func,solfs)
            
        # produce the internal deathrate functions
        def internal_death_rate_maker(key,func,solfs):
            def wrapper(field,t):
                sender=key[0]
                tss=field.tss
                loss=quad(func,t,t+tss)[0]
                stock=solfs[sender](t)
                relative_loss=loss/stock
                #print("int_relative_loss:=",relative_loss)
                dr=TsTpDeathRateField(relative_loss*np.ones(field.shape),tss)
                return(dr)
            return(wrapper)

        internal_death_rate_functions=dict()
        for key,func in mr.internal_flux_funcs().items():
            internal_death_rate_functions[key]=internal_death_rate_maker(key,func,solfs)


        # produce the external inputs
        def external_input_maker(receiver,func):
            def wrapper(t):
                return(quad(func,t,t+tss)[0])
            return(wrapper)
        
        external_inputs=dict()
        for receiver,func in mr.external_input_flux_funcs().items():
            external_inputs[receiver]=external_input_maker(receiver,func)
            
        start=times[0]
        age_dist_hist=TsTpMassFieldsPerPoolPerTimeStep.compute_from(
            initial_plains,
            external_inputs,
            internal_death_rate_functions,external_death_rate_functions,
            start,
            len(times)-1
        )    
        #age_dist_hist.single_pool_cartoon(0,"pool_0")
        fig = plt.figure()
        age_dist_hist.matrix_plot("plot_total_contents",fig)
        fig.savefig("total_content.pdf") #
        fig = plt.figure()
        #age_dist_hist.matrix_plot3d("plot_system_age_distributions_with_bins",fig)
        age_dist_hist.matrix_plot3d("plot_system_age_distributions_as_surfaces",fig)
        fig.savefig("system_age_distribution.pdf")
        fig = plt.figure()
        mr.plot_sols(fig)
        fig.savefig("mr_total_content.pdf")#compare
        #plt.close(fig) 
        
        
if __name__ == "__main__":
    unittest.main()
