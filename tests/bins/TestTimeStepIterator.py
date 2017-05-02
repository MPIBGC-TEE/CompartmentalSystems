# vim:set ff=unix expandtab ts=4 sw=4
import unittest
import numpy as np
import matplotlib.pyplot as plt
from CompartmentalSystems.bins.CompatibleTsTpMassFieldsPerPool import CompatibleTsTpMassFieldsPerPool
from CompartmentalSystems.bins.TimeStepIterator import TimeStepIterator
from CompartmentalSystems.bins.TimeStep import TimeStep
from CompartmentalSystems.bins.TsTpMassField import TsTpMassField 
from CompartmentalSystems.bins.TsTpDeathRateField import TsTpDeathRateField 

from testinfrastructure.InDirTest import InDirTest

class TestTimeStepIterator(InDirTest):
     def test_list_comprehension(self):
        ############################################################ 
        # get the components ready
        # - initial age distributions
        tss=1
        x,y=9,9
        s=(x,y)
        age_dist_0=TsTpMassField(np.zeros(s),tss)
        age_dist_0[2,2]=100
        
        x1,y1=4,4
        s=(x1,y1)
        age_dist_1=TsTpMassField(np.zeros(s),tss)
        age_dist_1[3,3]=100
        # the output from pool_1 has a bigger age_span than pool_0 can encompass
        # the code has to initialize pool_0 regarding the size of pool_1
        initial_plains=CompatibleTsTpMassFieldsPerPool([age_dist_0,age_dist_1])

        ############################################################ 
        # - deathrate functions
        loss_factor=0.1
        external_death_rate_funcs=dict()
        internal_death_rate_funcs=dict()
        def constant_well_mixed_death_rate(age_dist,t):
            # these functions must be able to define a field eta
            # of the same size as the age distribution it gets 
            # for all the ages present in age_dist it must
            # be able to compute the deathrate
            return(TsTpDeathRateField(loss_factor*np.ones(age_dist.arr.shape),age_dist.tss))

        external_death_rate_funcs[0]=constant_well_mixed_death_rate
        external_death_rate_funcs[1]=constant_well_mixed_death_rate
        internal_death_rate_funcs[(0,1)]=constant_well_mixed_death_rate
        # - input functions
        def zero_input(t):
            return(0)
        def const_input(t):
            return(5)
        external_input_funcs=dict()
        external_input_funcs[0]=const_input
        external_input_funcs[1]=zero_input

        drf=lambda t:0.2
        ############################################################ 
        # initialize the Iterator
        it=TimeStepIterator(
            initial_plains,
            external_input_funcs,
            internal_death_rate_funcs,
            external_death_rate_funcs,
            t0=5,
            number_of_steps=3
         )

        ############################################################ 
        ############################################################ 
        ############################################################ 
        # start testing
        # extract the complete information 
        steps=[ts for ts in it]

        # or only the part one is interested in 
        rectangles_for_first_pool=[ts.rectangles[0] for ts in it]
        #print("\n#####################################\nrectangles[0]",rectangles_for_first_pool)
        # or some parts 
        tuples=[(ts.time,ts.rectangles[0].total_content) for ts in it]
        x=[t[0] for t in tuples]
        y=[t[1] for t in tuples]
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.plot(x,y,"x")
        fig.savefig("plot.pdf")
        plt.close(fig.number)
        
     def test_from_ode_reservoir_model_run(self):
        pass

if __name__=="__main__":
    unittest.main()        
