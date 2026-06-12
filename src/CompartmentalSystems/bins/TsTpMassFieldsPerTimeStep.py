# vim: set ff=unix expandtab ts=4 sw=4:
import numpy as np
from .TsMassFieldsPerTimeStep import TsMassFieldsPerTimeStep
from .FieldsPerTimeStep import FieldsPerTimeStep


class TsTpMassFieldsPerTimeStep(FieldsPerTimeStep):
    @property
    def total_contents(self):
        return [field.total_content for field in self]

    def plot_total_contents(self, ax):
        ax.plot(self.times, self.total_contents)

    def system_age_distributions(self):
        systemAgeVectors = TsMassFieldsPerTimeStep(
            [field.sum_over_all_pool_ages() for field in self], self.start
        )
        return systemAgeVectors

    def plot_system_age_distributions_with_bins(self, ax, mr=None, pool=None):
        self.system_age_distributions().plot_bins(ax, mr, pool)

    def plot_system_age_distributions_as_surfaces(self, ax, mr=None, pool=None):
        self.system_age_distributions().plot_surface(ax, mr, pool)
