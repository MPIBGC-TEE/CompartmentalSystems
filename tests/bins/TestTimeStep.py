# vim:set ff=unix expandtab ts=4 sw=4
import unittest
from testinfrastructure.InDirTest import InDirTest
import numpy as np
from CompartmentalSystems.bins.TimeStep import TimeStep
from CompartmentalSystems.bins.TimeStepPlotter import TimeStepPlotter
from CompartmentalSystems.bins.TsTpMassField import TsTpMassField 
from CompartmentalSystems.bins.TsTpDeathRateField import TsTpDeathRateField 
from CompartmentalSystems.bins.CompatibleTsTpMassFieldsPerPool import CompatibleTsTpMassFieldsPerPool

class TestTimeStep(InDirTest):
    @unittest.skip("not implemented yet")
    def test_maxdeathrate(self):
        # If a pool has several outfluxes (e.g. 2 to other pools and one
        # outside) the "combined" deathrate of these 3 pipelines is
        # not allowed to be greater than 1.
        # Up to now we only check the single deathrates
        pass

    @unittest.skip("only in the meantime")
    def test_plot_single_pool_with_outflux(self):
        # get the components ready
        # - initial age distributions
        tss=1
        x,y=2,2
        s=(x,y)
        age_dist_0=TsTpMassField(3*np.ones(s),tss) #homogeneous age distribution
        
        # the output from pool_1 has a bigger age_span than pool_0 can encompass
        # the code has to initialize pool_0 regarding the size of pool_1
        initial_plains=CompatibleTsTpMassFieldsPerPool([age_dist_0])
        # no internal deathrate fields because there is no other pool
        # to transfer material to
        idr=dict()
        # external deathrate
        # to check the effect we only remove water that has been 
        # of one particular age bin.
        edr=dict()
        edr[0]=TsTpDeathRateField(np.zeros(s),tss) 
        edr[0][1,1]=0.5
        # - external input numbers
        ei=dict()
        ei[0]=2
        ts=TimeStep(0,initial_plains,idr,edr,ei)
        
        TimeStepPlotter(ts).plot_pdfs()
        refarr=np.zeros((x+1,y+1))
        refarr[2,2]=0.5 
        #the loss at 1,1 should have propagated in Ts and Tp        
        # by one bin widht tss

    @unittest.skip("only in the meantime")
    def test_plot_two_pool_with_internal_flux(self):
        # get the components ready
        # - initial age distributions
        tss=1
        x,y=4,4
        s=(x,y)
        age_dist_0=TsTpMassField(2*np.ones(s),tss)
        
        x1,y1=4,4
        s1=(x1,y1)
        age_dist_1=TsTpMassField(2*np.zeros(s1),tss)
        
        # the code has to initialize pool_0 regarding the size of pool_1
        initial_plains=CompatibleTsTpMassFieldsPerPool([age_dist_0,age_dist_1])
        # - deathrate fields
        # internal
        idr=dict()
        idr[(0,1)]=TsTpDeathRateField(np.zeros(s),tss) #one connection to second pool (well mixed=no preference for any age)
        idr[(0,1)][1,1]=0.5
        idr[(0,1)][2,2]=0.5
        idr[(0,1)][3,3]=0.5
        # external
        edr=dict()
        ei=dict()
        ei[0]=3
        ts=TimeStep(0,initial_plains,idr,edr,ei)
        TimeStepPlotter(ts).plot_pdfs()

    @unittest.skip("only in the meantime")
    def test_plot_three_pool(self):
        # get the components ready
        # - initial age distributions
        tss=1
        x,y=5,4
        s=(x,y)
        age_dist_0=TsTpMassField(np.zeros(s),tss)
        age_dist_0[3,3]=1
        age_dist_0[3,2]=2
        age_dist_0[2,2]=4
        
        x1,y1=5,4
        s1=(x1,y1)
        age_dist_1=TsTpMassField(np.zeros(s1),tss)
        age_dist_1[3,3]=0.5
        age_dist_1[3,2]=0.5
        age_dist_1[1,1]=0.5
        
        x2,y2=5,4
        s2=(x2,y2)
        age_dist_2=TsTpMassField(np.zeros(s2),tss)
        age_dist_2[3,0]=2
        age_dist_2[1,0]=4
        age_dist_2[0,0]=6
        # the code has to initialize pool_0 regarding the size of pool_1
        initial_plains=CompatibleTsTpMassFieldsPerPool([age_dist_0,age_dist_1,age_dist_2])
        # - deathrate fields
        # internal
        idr=dict()
        idr[(0,1)]=TsTpDeathRateField(1/8*np.ones(s),tss) #one connection to second pool (well mixed=no preference for any age)
        idr[(0,2)]=TsTpDeathRateField(1/8*np.ones(s),tss) #one connection to second pool (well mixed=no preference for any age)
        # external
        edr=dict()
        edr[0]=TsTpDeathRateField(1/4*np.ones(s),tss) # well mixed outflux regardless of age from pool 1
        edr[1]=TsTpDeathRateField(1/8*np.ones(s1),tss) # also from pool 2
        # - external input numbers
        ei=dict()
        ei[0]=3
        ei[1]=4
        ts=TimeStep(0,initial_plains,idr,edr,ei)

        ts.plot_pdfs()

if __name__=="__main__":
    unittest.main()        
