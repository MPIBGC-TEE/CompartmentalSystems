from unittest import TestCase
from CompartmentalSystems.InfiniteIterator import InfiniteIterator


class TestInfiniteIterator(TestCase):

    def test_init(self):
        v_0 = 0

        def f(i, n):
            return n+1

        itr = InfiniteIterator(start_value=v_0, func=f)
        res = [next(itr)for i in range(3)]

        self.assertEqual(res, [0, 1, 2])

    def test_init_with_max_iter(self):
        v_0 = 0

        def f(i, n):
            return n+1

        itr = InfiniteIterator(start_value=v_0, func=f, max_iter=3)
        res = [v for v in itr]
        self.assertEqual(res, [0, 1, 2])


