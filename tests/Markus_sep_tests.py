#!/usr/bin/env python3 
import unittest
from Test_smooth_reservoir_model import TestSmoothReservoirModel
from Test_smooth_model_run import TestSmoothModelRun
from Test_helpers_reservoir import TestHelpers_reservoir
from Test_start_distributions import TestStartDistributions
s=unittest.TestSuite()
#s.addTest(TestSmoothReservoirModel('test_jacobian'))
s.addTest(TestSmoothModelRun("test_age_densities"))
s.addTest(TestStartDistributions("test_numeric_staedy_state"))
#s.addTest(Test_start_distributions("test_compute_start_age_moments"))
unittest.TextTestRunner(verbosity=2).run(s)
