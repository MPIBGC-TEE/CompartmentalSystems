#!/usr/bin/env python3 
import unittest
from Test_smooth_reservoir_model import TestSmoothReservoirModel
from Test_smooth_model_run import TestSmoothModelRun
from Test_helpers_reservoir import TestHelpers_reservoir
s=unittest.TestSuite()
#s.addTest(TestSmoothReservoirModel('test_jacobian'))
s.addTest(TestSmoothModelRun("test_age_densities"))
#s.addTest(TestHelpers_reservoir("test_compute_start_age_moments"))
unittest.TextTestRunner(verbosity=2).run(s)
