import unittest
from testinfrastructure.InDirTest import InDirTest
import numpy as np
from sympy import symbols, Matrix

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel  

from CompartmentalSystems.pwc_model_run import PWCModelRun
from CompartmentalSystems.pwc_model_run_fd import PWCModelRunFD
from CompartmentalSystems.discrete_model_run_with_gross_fluxes import DiscreteModelRunWithGrossFluxes
from CompartmentalSystems.model_run import (
    plot_attributes
    ,plot_stocks_and_fluxes
)

class TestModelRun(InDirTest):
    def setUp(self):
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1,    1.5],
                    [ 0.5, -2  ]])
        u = Matrix(2, 1, [9,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        start_values = np.array([10,40])
        self.start_values = start_values
        self.t_0 = 0
        self.t_max = 10
        self.ntmo = 10
        self.fac = 2
        self.times=np.linspace(self.t_0,self.t_max,self.ntmo+1)

        self.pwc_mr = PWCModelRun(srm, {}, start_values, self.times)


    #@unittest.skip
    def test_DiscreteModelRunWithGrossFluxes_from_PWCModelRun(self):
        dmr = DiscreteModelRunWithGrossFluxes.from_PWCModelRun(self.pwc_mr)
        meths = [
            "solve"
            ,"acc_gross_external_input_vector"
            ,"acc_net_external_input_vector"
            ,"acc_gross_external_output_vector"
            ,"acc_net_external_output_vector"
            ,"acc_gross_internal_flux_matrix"
            ,"acc_net_internal_flux_matrix"
        ]
        for meth in meths:
            with self.subTest():
                 self.assertTrue(np.allclose(
                     getattr(self.pwc_mr, meth)()
                     ,getattr(dmr, meth)()
                 ))


    def test_net_vs_gross_for_different_time_steps(self):
        times_fine = np.linspace(self.t_0,self.t_max,self.fac*self.ntmo+1)
        times_extra_fine = np.linspace(self.t_0,self.t_max,self.fac**2*self.ntmo+1) 
        pwc_mr=self.pwc_mr
        
        pwc_mr_fine = PWCModelRun(
            pwc_mr.model
            ,pwc_mr.parameter_dict 
            ,pwc_mr.start_values
            ,times_fine
            ,pwc_mr.func_set
        )
        pwc_mr_extra_fine = PWCModelRun(
            pwc_mr.model
            ,pwc_mr.parameter_dict 
            ,pwc_mr.start_values
            ,times_extra_fine
            ,pwc_mr.func_set
        )
        # We build a discrete model where we use the gross fluxes
        # as arguments for BOTH (net and gross) fluxes.
        # This simulates the real world scenario.
        # Since the net fluxes are different from the gross fluxes
        # the discrete model assumes wrong net fluxes.
        # The correct values would be given by pwc_mr_fine.
        # For a smaller step size the gross fluxes would be essentially
        # the same (interpolating original ones) but the difference
        # to the net fluxes would be smaller, since the latter approach 
        # the gross fluxes in the limit of small time steps.
        # So the bigger the time step the bigger the error in the 
        # net fluxes and hence the reconstruction of the discrete Bs.
        xs_fine, net_Us_fine, net_Fs_fine, net_Rs_fine = self.pwc_mr.fake_net_discretized_output(times_fine)
        xs_fine, gross_Us_fine, gross_Fs_fine, gross_Rs_fine = self.pwc_mr.fake_gross_discretized_output(times_fine)
        dmr_wrong_fine = DiscreteModelRunWithGrossFluxes.reconstruct_from_fluxes_and_solution(
           times_fine,
           xs_fine,
           net_Us = gross_Us_fine,
           net_Fs = gross_Fs_fine,
           net_Rs = gross_Rs_fine,
           gross_Us = gross_Us_fine,
           gross_Fs = gross_Fs_fine,
           gross_Rs = gross_Rs_fine
        )
        plot_stocks_and_fluxes(
            [
                self.pwc_mr
                ,pwc_mr_fine
                ,pwc_mr_extra_fine
                ,dmr_wrong_fine
            ]
            ,'stocks_and_fluxes.pdf'
            , labels = ['mr_normal','mr_fine','mr_extra_fine','dmr_wrong_fine']
        )       

    #@unittest.skip
    def test_DiscreteModelRunFromFakeData(self):
        times = self.pwc_mr.times
        xs, net_Us, net_Fs, net_Rs = self.pwc_mr.fake_net_discretized_output(times)
        xs, gross_Us, gross_Fs, gross_Rs = self.pwc_mr.fake_gross_discretized_output(times)
        
        dmr = DiscreteModelRunWithGrossFluxes.reconstruct_from_fluxes_and_solution(
           times,
           xs,
           net_Us,
           net_Fs,
           net_Rs,
           gross_Us,
           gross_Fs,
           gross_Rs
        )
        meths = [
            "solve"
            ,"acc_gross_external_input_vector"
            ,"acc_net_external_input_vector"
            ,"acc_gross_external_output_vector"
            ,"acc_net_external_output_vector"
            ,"acc_gross_internal_flux_matrix"
            ,"acc_net_internal_flux_matrix"
        ]
        for meth in meths:
            with self.subTest():
                 self.assertTrue(np.allclose(
                     getattr(self.pwc_mr, meth)()
                     ,getattr(dmr, meth)()
                 ))

    @unittest.skip
    def test_PWCRunFD(self):
        times = self.pwc_mr.times
        xs, gross_Us, gross_Fs, gross_Rs = self.pwc_mr.fake_gross_discretized_output(times)
        
        pwc_mr_fd = PWCModelRunFD(
            self.pwc_mr.model.time_symbol
            ,times
            ,self.pwc_mr.start_values
            ,gross_Us
            ,gross_Fs
            ,gross_Rs
        )
        meths = [
            "solve"
            ,"acc_gross_external_input_vector"
            ,"acc_net_external_input_vector"
            ,"acc_gross_external_output_vector"
            ,"acc_net_external_output_vector"
            ,"acc_gross_internal_flux_matrix"
            ,"acc_net_internal_flux_matrix"
        ]
        for meth in meths:
            with self.subTest():
                 ref = getattr(self.pwc_mr, meth)()
                 res = getattr(pwc_mr_fd, meth)()
                 self.assertTrue(np.allclose(
                     ref 
                     ,res
                     ,rtol = 3e-02 
                     # For this linear constant model 
                     # the error should actually be zero 
                     # and is only due to numerical inaccuracy. 
                 ))
        plot_stocks_and_fluxes(
            [
                self.pwc_mr
                ,pwc_mr_fd
            ],
            'stocks_and_fluxes.pdf'
        )       
                 
