#!/usr/bin/env python3
# vim:set ff=unix expandtab ts=4 sw=4:
# this is a pure python version 
# run with pyhton3 run_tests.py in a venv

from concurrencytest import ConcurrentTestSuite, fork_for_tests
import unittest
import sys
from pathlib import Path

def main():
    # the following monkeypatching of the path would 
    # make the execution in subdirectories impossible 
    #print("\n###################### running single tests ##########################\n")
    #s1=unittest.TestSuite()
    #from Test_smooth_model_run import TestSmoothModelRun
    #s1.addTest(TestSmoothModelRun('test_linearize'))
    #r = unittest.TextTestRunner()
    #res = r.run(s1)
    #if len(res.errors) + len(res.failures) > 0:
    #    sys.exit(1)

    print("\n###################### running tests ##########################\n")

    s = unittest.defaultTestLoader.discover('', pattern='Test*')
    #p = unittest.defaultTestLoader.discover('', pattern='Pinned_Test*')
    #s.addTests(p)
    #concurrent_suite = s
    concurrent_suite = ConcurrentTestSuite(s, fork_for_tests(64))
    r = unittest.TextTestRunner()

    res = r.run(concurrent_suite)
    if len(res.errors) + len(res.failures) > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
