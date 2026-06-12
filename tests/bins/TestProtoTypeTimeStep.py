# vim:set ff=unix expandtab ts=4 sw=4
# There is a TestTimeStep for the full implementation

from copy import deepcopy
import matplotlib

matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import unittest
from testinfrastructure.InDirTest import InDirTest


class Content:
    def __init__(self, amount):
        self.val = amount


class TimeStep:
    def __init__(self, time, content, death_rate_num):
        self.time = time
        self.content = content
        self.death_rate_num = death_rate_num

        self.loss = self.death_rate_num * self.content.val
        self.gain = 0.1  # or computed somehow

    @property
    def updated_content(self):
        val = self.content.val + self.gain - self.loss
        return Content(val)


class TimeStepIterator:
    """iterator for looping over the results of a difference equation"""

    def __init__(self, drf, t0, c0, tss, number_of_steps):
        self.t0 = t0
        self.c0 = c0
        self.tss = tss
        self.number_of_steps = number_of_steps
        self.deathrate_func = drf
        self.reset()

    def reset(self):
        self.i = 0
        self.time = self.t0
        self.content = self.c0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        number_of_steps = self.number_of_steps
        if self.i == number_of_steps:
            raise StopIteration
        drf = self.deathrate_func
        t = self.t0 + self.i * self.tss
        dr = drf(t)
        ts = TimeStep(t, self.content, dr)
        self.content = ts.updated_content
        self.i += 1
        return ts


class TestTimeStepIterator(InDirTest):
    def test_list_comprehension(self):
        drf = lambda t: 0.2
        it = TimeStepIterator(drf, t0=5, c0=Content(10), tss=0.01, number_of_steps=30)
        # extract the complete information
        steps = [ts for ts in it]
        it.i = 0
        # or only the part one is interested in
        c_of_t = [ts.content.val for ts in it]
        # or some parts
        tuples = [(ts.time, ts.content.val) for ts in it]
        x = [t[0] for t in tuples]
        y = [t[1] for t in tuples]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y, "x")
        fig.savefig("plot.pdf")
        plt.close(fig.number)


if __name__ == "__main__":
    unittest.main()
