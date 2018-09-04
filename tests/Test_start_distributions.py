
# vim:set ff=unix expandtab ts=4 sw=4:
from concurrencytest import ConcurrentTestSuite, fork_for_tests
import sys
import unittest
import plotly.graph_objs as go
from plotly.offline import plot
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol,Matrix, symbols, sin, Piecewise, DiracDelta, Function
from CompartmentalSystems.helpers_reservoir import factor_out_from_matrix, parse_input_function, melt, MH_sampling, stride, is_compartmental, func_subs, numerical_function_from_expression,pe
from CompartmentalSystems.start_distributions import \
    start_age_moments_from_empty_spin_up, \
    start_age_moments_from_steady_state, \
    compute_fixedpoint_numerically,\
    start_age_distributions_from_steady_state, \
    start_age_distributions_from_empty_spinup, \
    start_age_distributions_from_zero_initial_content
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from testinfrastructure.InDirTest import InDirTest

class TestStartDistributions(InDirTest):
    def test_start_age_distribuions_from_empty_spinup(self):
        # two-dimensional nonlinear 
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        t = Symbol('t')
        
        f_expr = Function('f')(t)
        
        
        input_fluxes = {0: C_0*f_expr+2, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {(0,1):0.5*C_0**3}
        srm = SmoothReservoirModel(state_vector, t, input_fluxes, output_fluxes, internal_fluxes)
        
        parameter_set={}
        def f_func( t_val):
            return np.sin(t_val)+1.0
        
        func_set = {f_expr: f_func}

        t_min = 0
        t_max = 2*np.pi
        n_steps=101
        times = np.linspace(t_min,t_max,n_steps) 
        # create a model run that starts with all pools empty
        smr = SmoothModelRun(srm, parameter_set=parameter_set, start_values=np.zeros(srm.nr_pools), times=times,func_set=func_set)
        # choose a t_0 somewhere in the times
        t0_index = int(n_steps/2)
        t0       = times[t0_index]
        a_dens_func_t0,pool_contents=start_age_distributions_from_empty_spinup(srm,t0=t0,parameter_set=parameter_set,func_set=func_set)
        pe('pool_contents',locals())
        
        # construct a function p that takes an age array "ages" as argument
        # and gives back a three-dimensional ndarray (ages x times x pools)
        # from the a array-valued function representing the start age density
        p=smr.pool_age_densities_func(start_age_distributions_from_zero_initial_content(srm))

        # for this particular example we are only interrested in ages that are smaller than t_max
        # the particular choice ages=times means that t_0_ind is the same in both arrays
        ages=times 
        t0_age_index=t0_index

        pool_dens_data=p(ages)
        for n in range(srm.nr_pools):

            fig=smr.plot_3d_density_plotly("pool {0}".format(n),pool_dens_data[:,:,n],ages)
            # plot the computed start age density for t0 on top
            trace_on_surface = go.Scatter3d(
                x=np.array([-t0 for a in ages]),
                y=np.array([a for a in ages]),
                z=np.array([a_dens_func_t0(a)[n] for a in ages]),
                mode = 'lines',
                line=dict(
                    color='#FF0000',
                    width=15
                    )
                #,
                #showlegend = legend_on_surface
            )
            #smr.add_equilibrium_surface_plotly(fig)
            fig['data'] += [trace_on_surface]
            plot(fig,filename="test_{0}.html".format(n),auto_open=False)
           
            # make sure that the values for the model run at t0 conince with the values computed by the             # function returned by the function under test
            res_data=np.array([a_dens_func_t0(a)[n] for a in ages])
            ref_data=pool_dens_data[:,t0_index,n]
            self.assertTrue(np.allclose(res_data,ref_data,rtol=1e-3))

            # make sure that the density is zero for all values of age bigger than t0
            self.assertTrue(np.all(res_data[t0_age_index:]==0))


        #compute the start age distribution
    def test_start_age_distribuions_from_steady_state(self):
        # two-dimensional non-autonomous linear 
        # we can ommit the initial guess for the fixedpoint
        # since the function can employ an explicit formula
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        t = Symbol('t')
        
        f_expr = Function('f')(t)
        def f_func( t_val):
            return np.sin(t_val)+1.0
        
        func_set = {f_expr: f_func}
        
        
        input_fluxes = {0: C_0*f_expr+2, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, t, input_fluxes, output_fluxes, internal_fluxes)
        t0=3/2*np.pi
        parameter_set={}

        #compute the start age distribution
        a_dens_function,x_fix = start_age_distributions_from_steady_state(srm,t0=t0,parameter_set={},func_set=func_set)
        # create a model run that starts at x_fix and t0
        times = np.linspace(t0, 8*np.pi,41)
        smr = SmoothModelRun(srm, parameter_set=parameter_set, start_values=x_fix, times=times,func_set=func_set)

        # construct a function p that takes an age array "ages" as argument
        # and gives back a three-dimensional ndarray (ages x times x pools)
        # from the a array-valued function of a single age a_dens_function
        p=smr.pool_age_densities_func(a_dens_function)
        ages=np.linspace(0,3,31)
        pool_dens_data=p(ages)
        system_dens_data=smr.system_age_density(pool_dens_data)
        fig=smr.plot_3d_density_plotly('pool 1',pool_dens_data[:,:,0],ages)
        # two-dimensional non-autonomous nonlinear 
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        t = Symbol('t')
        
        f_expr = Function('f')(t)
        def f_func( t_val):
            return np.sin(t_val)+1.0
        
        func_set = {f_expr: f_func}
        
        
        input_fluxes = {0: C_0*f_expr+2, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {(0,1):0.5*C_0**3}
        srm = SmoothReservoirModel(state_vector, t, input_fluxes, output_fluxes, internal_fluxes)
        t0=3/2*np.pi
        x0=np.array([1,2])
        parameter_set={}

        #compute the start age distribution
        a_dens_function,x_fix = start_age_distributions_from_steady_state(srm,t0=t0,parameter_set={},func_set=func_set,x0=np.array([1,2]))
        # create a model run that starts at x_fix and t0
        times = np.linspace(t0, 8*np.pi,41)
        smr = SmoothModelRun(srm, parameter_set=parameter_set, start_values=x_fix, times=times,func_set=func_set)

        # construct a function p that takes an age array "ages" as argument
        # and gives back a three-dimensional ndarray (ages x times x pools)
        # from the a array-valued function of a single age a_dens_function
        p=smr.pool_age_densities_func(a_dens_function)
        ages=np.linspace(0,3,31)
        pool_dens_data=p(ages)
        system_dens_data=smr.system_age_density(pool_dens_data)
        fig=smr.plot_3d_density_plotly('pool 1',pool_dens_data[:,:,0],ages)
        trace_on_surface = go.Scatter3d(
            #name=name,
            #x=-strided_times, y=strided_data, z=strided_z,
            #x=[-times[5:10]],
            #y=ages,
            x=np.array([-times[0] for a in ages]),
            y=np.array([a for a in ages]),
            z=np.array([a_dens_function(a)[0] for a in ages]),
            #z=np.array([2 for a in ages]),
            mode = 'lines',
            line=dict(
                color='#FF0000',
                width=15
                )
            #,
            #showlegend = legend_on_surface
        )
        smr.add_equilibrium_surface_plotly(fig)
        fig['data'] += [trace_on_surface]
        plot(fig,filename='test.html',auto_open=False)


    def test_numeric_staedy_state(self):
        # two-dimensional nonlinear 
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        t = Symbol('t')

        input_fluxes = {0: 4 , 1: 2}
        output_fluxes = {0: C_0**2, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, t, input_fluxes, output_fluxes, internal_fluxes)
        res=compute_fixedpoint_numerically(srm,t0=0,x0=np.array([1,1]),parameter_set={},func_set={})
        ref=np.array([2,2])
        self.assertTrue(np.allclose(res,ref))
        self.assertTupleEqual(res.shape,ref.shape)
       

        # two-dimensional with external functions
        # although linear the code will assume u(C_1,C_2,t) to be nonlinear since it can not check
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        t = Symbol('t')


        f_expr = Function('f')(C_0, t)
        def f_func(C_0_val,  t_val):
            return C_0_val+t_val
        
        func_set = {f_expr: f_func}


        input_fluxes = {0: f_expr, 1: 2}
        output_fluxes = {0: 2*C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, t, input_fluxes, output_fluxes, internal_fluxes)
        res=compute_fixedpoint_numerically(srm,t0=2,x0=np.array([4,4]),parameter_set={},func_set=func_set)
        ref=np.array([2,2])
        self.assertTrue(np.allclose(res,ref))
        self.assertTupleEqual(res.shape,ref.shape)

        # two-dimensional with coupled with linear external functions 
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        t = Symbol('t')


        f_expr = Function('f')(t)
        def f_func( t_val):
            return np.sin(t_val)
        
        func_set = {f_expr: f_func}


        input_fluxes = {0: C_0*f_expr, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {(0,1):0.5*C_0**3}
        srm = SmoothReservoirModel(state_vector, t, input_fluxes, output_fluxes, internal_fluxes)
        t0=2
        res=compute_fixedpoint_numerically(srm,t0=t0,x0=np.array([1,2]),parameter_set={},func_set=func_set)
        # make sure that the righthandside of the ode is zero
        F_sym=srm.F
        F_func=numerical_function_from_expression(F_sym,tup=(C_0,C_1,t),parameter_set={},func_set=func_set)
        F_res=F_func(*res,t0)
        ref=np.array([0,0])
        self.assertTrue(np.allclose(F_res,ref))
        self.assertTupleEqual(res.shape,ref.shape)
        



    def test_compute_start_age_moments_from_steady_state(self):
        # two-dimensional linear autonomous
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        t = Symbol('t')
        input_fluxes = {0: 1, 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, t, input_fluxes, output_fluxes, internal_fluxes)
        age_moment_vector=start_age_moments_from_steady_state(srm,t0=0,parameter_set={},func_set={},max_order=2)
        self.assertEqual(age_moment_vector.shape,(2,2))
        # we only check the #expectation values since B is the identity the number are the same as in the input fluxes 
        ref_ex=np.array([1,2]) 
        for pool in range(srm.nr_pools):
            self.assertTrue(np.allclose(age_moment_vector[:,pool], ref_ex))
        
        # two-dimensional linear non-autonomous
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        t = Symbol('t')
        input_fluxes = {0: 1*(sin(t)+1), 1: 2}
        output_fluxes = {0: C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, t, input_fluxes, output_fluxes, internal_fluxes)
        age_moment_vector=start_age_moments_from_steady_state(srm,t0=0,parameter_set={},func_set={},max_order=2)
        self.assertEqual(age_moment_vector.shape,(2,2))
        # we only check the #expectation values since B is the identity the number are the same as in the input fluxes 
        ref_ex=np.array([1,2]) 
        for pool in range(srm.nr_pools):
            self.assertTrue(np.allclose(age_moment_vector[:,pool], ref_ex))
        
        # two-dimensional linear but state dependent input non-autonomous 
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        t = Symbol('t')
        input_fluxes = {0: 0.5*C_0+sin(t)+1, 1: 2}
        output_fluxes = {0: 1.5*C_0, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, t, input_fluxes, output_fluxes, internal_fluxes)
        age_moment_vector=start_age_moments_from_steady_state(srm,t0=0,parameter_set={},func_set={},max_order=2)
        self.assertEqual(age_moment_vector.shape,(2,2))
        # we only check the #expectation values since B is the identity the number are the same as in the input fluxes 
        ref_ex=np.array([1,2]) 
        for pool in range(srm.nr_pools):
            self.assertTrue(np.allclose(age_moment_vector[:,pool], ref_ex))

        # two-dimensional nonlinear state dependent input non-autonomous 
        C_0, C_1 = symbols('C_0 C_1')
        state_vector = [C_0, C_1]
        t = Symbol('t')
        input_fluxes = {0: 0.5*C_0+sin(t)+1, 1: 2}
        output_fluxes = {0: 1.5*C_0**3, 1: C_1}
        internal_fluxes = {}
        srm = SmoothReservoirModel(state_vector, t, input_fluxes, output_fluxes, internal_fluxes)
        age_moment_vector=start_age_moments_from_steady_state(srm,t0=0,parameter_set={},func_set={},max_order=2)
        self.assertEqual(age_moment_vector.shape,(2,2))
        # we only check the #expectation values since B is the identity the number are the same as in the input fluxes 
        #ref_ex=np.array([1,2]) 
        #for pool in range(srm.nr_pools):
        #    self.assertTrue(np.allclose(age_moment_vector[:,pool], ref_ex))




if __name__ == '__main__':
    suite=unittest.defaultTestLoader.discover(".",pattern=__file__)
    # Run same tests across 16 processes
    concurrent_suite = ConcurrentTestSuite(suite, fork_for_tests(16))
    runner = unittest.TextTestRunner()
    res=runner.run(concurrent_suite)
    # to let the buildbot fail we set the exit value !=0 if either a failure or 
    # error occurs
    if (len(res.errors)+len(res.failures))>0:
        sys.exit(1)
