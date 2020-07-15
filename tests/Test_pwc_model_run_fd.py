import unittest

import numpy as np
import xarray as xr
from sympy import Function, Matrix, symbols

from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
from CompartmentalSystems.smooth_model_run import SmoothModelRun
from CompartmentalSystems.pwc_model_run_fd import PWCModelRunFD

from bgc_md2.models.CARDAMOM.CARDAMOMlib import load_mdo

import os.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestPWCModelRunFD(unittest.TestCase):

    def setUp(self):
        x, y, t, k = symbols("x y t k")
        u_1 = Function('u_1')(x, t)
        state_vector = Matrix([x, y])
        B = Matrix([[-1,  1.5],
                    [k/(t+1), -2]])
        u = Matrix(2, 1, [u_1, 1])
        self.srm = SmoothReservoirModel.from_B_u(
            state_vector,
            t,
            B,
            u
        )

        start_values = np.array([10, 40])
        t_0 = 0
        t_max = 10
        times = np.linspace(t_0, t_max, 11)

        parameter_dict = {k: 1}
        func_dict = {u_1: lambda x, t: 9}

        self.smr = SmoothModelRun(
            self.srm,
            parameter_dict,
            start_values,
            times,
            func_dict
        )
            

    def test_reconstruction_accuracy(self):
        smr = self.smr
        xs, gross_Us, gross_Fs, gross_Rs =\
            smr.fake_gross_discretized_output(smr.times)

        pwc_mr_fd = PWCModelRunFD(
            smr.model.time_symbol,
            smr.times,
            xs[0, :],
            gross_Us,
            gross_Fs,
            gross_Rs
        )

        self.assertTrue(
            np.allclose(
                smr.solve(),
                pwc_mr_fd.solve(),
                rtol=1e-03
            )
        )

        self.assertTrue(
            np.allclose(
                smr.acc_gross_external_input_vector(),
                pwc_mr_fd.acc_gross_external_input_vector(),
                rtol=1e-4
            )
        )

        self.assertTrue(
            np.allclose(
                smr.acc_gross_external_output_vector(),
                pwc_mr_fd.acc_gross_external_output_vector(),
                rtol=1e-4
            )
        )

        self.assertTrue(
            np.allclose(
                smr.acc_gross_internal_flux_matrix(),
                pwc_mr_fd.acc_gross_internal_flux_matrix(),
                rtol=1e-4
            )
        )

    def test_get_stock(self):
        filename = 'cardamom_for_holger_10_ensembles.nc'
        my_data_path = os.path.join(THIS_DIR, filename)
        dataset = xr.open_dataset(my_data_path)
        ds = dataset.isel(ens=0, lat=0, lon=0, time=slice(None, 24))
        mdo = load_mdo(ds)
        mr = mdo.create_model_run()

        soln = mr.solve()
        for nr, pool_dict in enumerate(mdo.model_structure.pool_structure):
            with self.subTest():
                pool_name = pool_dict['pool_name']
                self.assertTrue(
                    np.all(soln[:, nr] == mr.get_stock(mdo, pool_name).data)
                ) 

    def test_get_acc_gross_external_input_flux(self):
        filename = 'cardamom_for_holger_10_ensembles.nc'
        my_data_path = os.path.join(THIS_DIR, filename)
        dataset = xr.open_dataset(my_data_path)
        ds = dataset.isel(ens=0, lat=0, lon=0, time=slice(None, 24))
        mdo = load_mdo(ds)
        mr = mdo.create_model_run()

        ms = mdo.model_structure
        Us = mr.acc_gross_external_input_vector()
        for nr, pool_dict in enumerate(ms.pool_structure):
            with self.subTest():
                pool_name = pool_dict['pool_name']
                if pool_name in ms.external_input_structure.keys():
                    self.assertTrue(
                        np.all(Us[:, nr] ==\
                               mr.get_acc_gross_external_input_flux(
                                   mdo, 
                                   pool_name
                               ).data
                        )
                    )
                else:
                    self.assertTrue(np.all(Us[:, nr] == 0))
                
    def test_get_acc_gross_external_output_flux(self):
        filename = 'cardamom_for_holger_10_ensembles.nc'
        my_data_path = os.path.join(THIS_DIR, filename)
        dataset = xr.open_dataset(my_data_path)
        ds = dataset.isel(ens=0, lat=0, lon=0, time=slice(None, 24))
        mdo = load_mdo(ds)
        mr = mdo.create_model_run()

        ms = mdo.model_structure
        Rs = mr.acc_gross_external_output_vector()
        for nr, pool_dict in enumerate(ms.pool_structure):
            with self.subTest():
                pool_name = pool_dict['pool_name']
                if pool_name in ms.external_output_structure.keys():
                    self.assertTrue(
                        np.all(Rs[:, nr] ==\
                               mr.get_acc_gross_external_output_flux(
                                   mdo, 
                                   pool_name
                               ).data
                        )
                    )
                else:
                    self.assertTrue(np.all(Rs[:, nr] == 0))
                
    def test_get_acc_gross_internal_flux(self):
        filename = 'cardamom_for_holger_10_ensembles.nc'
        my_data_path = os.path.join(THIS_DIR, filename)
        dataset = xr.open_dataset(my_data_path)
        ds = dataset.isel(ens=0, lat=0, lon=0, time=slice(None, 24))
        mdo = load_mdo(ds)
        mr = mdo.create_model_run()

        ms = mdo.model_structure
        Fs = mr.acc_gross_internal_flux_matrix()
        for nr_from, pool_dict_from in enumerate(ms.pool_structure):
            pool_name_from = pool_dict_from['pool_name']
            for nr_to, pool_dict_to in enumerate(ms.pool_structure):
                pool_name_to = pool_dict_to['pool_name']
                with self.subTest():
                    if (pool_name_from, pool_name_to) in\
                        ms.horizontal_structure.keys():
                        self.assertTrue(
                            np.all(Fs[:, nr_to, nr_from] ==\
                                   mr.get_acc_gross_internal_flux(
                                       mdo, 
                                       pool_name_from,
                                       pool_name_to
                                   ).data
                            )
                        )
                    else:
                        self.assertTrue(np.all(Fs[:, nr_to, nr_from] == 0))
                
###############################################################################


if __name__ == '__main__':
    unittest.main()
