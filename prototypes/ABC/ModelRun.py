import unittest 
from numpy import NaN

from abc import ABCMeta, abstractmethod


class ModelRun(metaclass=ABCMeta):
    # abstractmehtods HAVE to be overloaded in the subclasses
    # the decorator should only be used inside a class definition
    @abstractmethod
    def solve(self): 
        return NaN
 
    
    # non abstract methods could be implemented and would
    # then be inherited automatically by subclasses
    # BUT do not have to be overloaded
    def do_some_thing(self):
        return 43



class ModelRunWithMissingMethods(ModelRun):
    # does not implement solve yet
    pass

class PWS_ModelRun(ModelRun):
    def solve(self): 
        return 42

class D_ModelRun(ModelRun):
    def solve(self): 
        return 24


class TestModelRun(unittest.TestCase):
    def test_init__(self):
        # The abstract class itself can not be instanciated
        with self.assertRaises(TypeError):
            ModelRun()  

        # Subclasses of the abstract class itself have to implement ALL abstractmethods 
        with self.assertRaises(TypeError):
            ModelRunWithMissingMethods()  

        # A subclass implementing the abstract methods can be instanciated
        mr=PWS_ModelRun()


    def test_do_some_thing(self):
        mr1=PWS_ModelRun()
        self.assertEqual(mr1.solve(),42)
        self.assertEqual(mr1.do_some_thing(),43)

        mr2=D_ModelRun()
        self.assertEqual(mr2.solve(),24)
        self.assertEqual(mr2.do_some_thing(),43)

if __name__ == '__main__':
    unittest.main()
