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
    ,plot_stocks_and_net_fluxes
    ,plot_stocks_and_gross_fluxes
    ,plot_stocks_and_fluxes
)

class TestModelRun(InDirTest):
    def setUp(self):
        x, y, t = symbols("x y t")
        state_vector = Matrix([x,y])
        B = Matrix([[-1,    2],
                    [ 0.5, -2]])
        u = Matrix(2, 1, [9,1])
        srm = SmoothReservoirModel.from_B_u(state_vector, t, B, u)

        start_values = np.array([1,1])
        self.start_values = start_values
        t_0 = 0
        t_max = 10
        ntmo = 10
        fac = 2
        times=np.linspace(t_0,t_max,ntmo+1)
        self.times=times
        times_fine = np.linspace(t_0,t_max,fac*ntmo+1)
        times_extra_fine = np.linspace(t_0,t_max,fac**2*ntmo+1) 

        self.pwc_mr = PWCModelRun(srm, {}, start_values, times)
        self.pwc_mr_fine = PWCModelRun(srm, {}, start_values, times_fine)
        self.pwc_mr_extra_fine = PWCModelRun(srm, {}, start_values, times_extra_fine)

    @unittest.skip
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

    def test_DiscretizationError(self):
        #self.pwc_mr_fine = PWCModelRun(srm, {}, start_values, times_fine)
        raise

    def test_net_vs_gross_for_different_time_steps(self):
        #self.pwc_mr_fine = PWCModelRun(srm, {}, start_values, times_fine)
        #times=self.pwc_mr.times
        #times_fine=self.pwc_mr_fine.times
        #xs, gross_Us, gross_Fs, gross_Rs = self.pwc_mr.fake_gross_discretized_output(times)
        #xs_fine, gross_Us_fine, gross_Fs_fine, gross_Rs_fine = self.pwc_mr.fake_gross_discretized_output(times_fine)
        # build discrete models by abusing the gross fluxes
        # as net fluxes
        #dmr_wrong = DiscreteModelRunWithGrossFluxes.reconstruct_from_fluxes_and_solution(
        #   times,
        #   xs,
        #   net_Us = gross_Us,
        #   net_Fs = gross_Fs,
        #   net_Rs = gross_Rs,
        #   gross_Us = gross_Us,
        #   gross_Fs = gross_Fs,
        #   gross_Rs = gross_Rs
        #)
        #dmr_wrong_fine = DiscreteModelRunWithGrossFluxes.reconstruct_from_fluxes_and_solution(
        #   times_fine,
        #   xs_fine,
        #   net_Us = gross_Us_fine,
        #   net_Fs = gross_Fs_fine,
        #   net_Rs = gross_Rs_fine,
        #   gross_Us = gross_Us_fine,
        #   gross_Fs = gross_Fs_fine,
        #   gross_Rs = gross_Rs_fine
        #)
        plot_stocks_and_fluxes(
            [
                self.pwc_mr
                ,self.pwc_mr_fine
                ,self.pwc_mr_extra_fine
                #,dmr_wrong
                #,dmr_wrong_fine
            ]
            ,'stocks_and_fluxes.pdf'
            , labels = ['mr_normal','mr_fine','mr_extra_fine']
        )       

    @unittest.skip
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
            ,xs
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
                 self.assertTrue(np.allclose(
                     getattr(self.pwc_mr, meth)()
                     ,getattr(pwc_mr_fd, meth)()
                     ,rtol = 3.75e-02 
                 ))
        plot_stocks_and_net_fluxes(
            [
                self.pwc_mr
                ,pwc_mr_fd
            ],
            'stocks_and_net_fluxes.pdf'
        )       
        plot_stocks_and_gross_fluxes(
            [
                self.pwc_mr
                ,pwc_mr_fd
            ],
            'stocks_and_gross_fluxes.pdf'
        )       
                 
