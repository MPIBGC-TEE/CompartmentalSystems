# vim: set ff=unix expandtab ts=4 sw=4:
import numpy as np
from sympy import latex
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from .density_algorithm import losses, gains
from .TsTpMassField import TsTpMassField
from .TsTpMassFieldsPerTimeStep import TsTpMassFieldsPerTimeStep


class TsTpMassFieldsPerPoolPerTimeStep:
    # this class represents the results of the simulations for a multipool reservoir model
    # for each system_age
    #     for each pool_age
    #        for each pool
    #            for each timestep
    #                the amount of mass is stored

    @classmethod
    def from_time_step_iterator(cls, iterator):
        l = [ts.rectangles for ts in iterator]
        start = iterator.t0
        obj = cls(l, start)
        return obj

    @property
    def tss(self):
        return self[0][0].tss

    def __init__(self, multi_pool_pyramid, start):
        # can be replaced later by something smarter
        self.multi_pool_pyramid = multi_pool_pyramid
        self.start = start

    #    def movie(self,tss,pool_number,trunk):
    #        movie_file=trunk+".mp4"
    #        pyr=self.multi_pool_pyramid
    #        max_shape=pyr[-1][pool_number].shape
    #
    #        FFMpegWriter = manimation.writers['ffmpeg']
    #        metadata = dict(title='', artist='The TEE Group',
    #                        comment='')
    #        writer = FFMpegWriter(fps=1, metadata=metadata)
    #
    #        fig = plt.figure()
    #        #ax=fig.add_subplot(1,1,1,projection="3d")
    #        nots=len(pyr)
    #        with writer.saving(fig, movie_file,100, nots):
    #            for i in range(nots):
    #                fig.clf()
    #                ax=fig.add_subplot(1,1,1,projection="3d")
    #                ss=TsTpMassField(pyr[i][pool_number],self.tss)
    #                ss.plot(ax,max_shape)
    #                writer.grab_frame()

    @property
    def number_of_pools(self):
        return self.multi_pool_pyramid[0].number_of_pools

    def singlePoolPyramid(self, pool_nr):
        sppyr = [fields[pool_nr] for fields in self.multi_pool_pyramid]
        return TsTpMassFieldsPerTimeStep(sppyr, self.start)

    def matrix_plot(self, sub_func_name, fig):
        plt.figure(fig.number)  # activate the figure
        # creates a matrix plot with the plot for the pools on the diagonal
        # and the plots for the pipelines in the off diagonal parts as functions of time
        # and add them
        n = self.number_of_pools
        # diagonal entries
        for i in range(n):
            ax = plt.subplot2grid((n, n), (i, i))
            ax.set_title(sub_func_name + " pool " + str(i))
            pool_pyr = self.singlePoolPyramid(i)
            getattr(pool_pyr, sub_func_name)(ax)

        # nondiagonal entries
        # not implemented yet
        # get the fluxes

    # fixme: treatment of title, state_variables,time_symbol
    # since mr specific knowledge does not want to live here at all
    def matrix_plot3d(self, sub_func_name, fig, title=None, mr=None):
        # creates a matrix plot with for each pool
        # and adds them
        n = self.number_of_pools
        # holger: here you can change which pools are plotted
        # n=1
        fig.set_figheight(n * fig.get_figwidth())
        # diagonal entries
        for i in range(n):
            ax = fig.add_subplot(n, 1, i + 1, projection="3d")
            ax.view_init(elev=15, azim=72)

            if title:
                sv_str = "$" + latex(mr.model.state_variables[i]) + "$"
                ax.set_title(title + " of " + sv_str, fontsize=20)
            else:
                ax.set_title(sub_func_name + " pool " + str(i + 1), fontsize=20)

            pool_pyr = self.singlePoolPyramid(i)
            # fixme: reservoir model to next plot method
            getattr(pool_pyr, sub_func_name)(ax, mr, i)

        # nondiagonal entries
        # not implemented yet
        # get the fluxes

    def single_pool_cartoon(self, pool_number, trunk):
        pyr = self.multi_pool_pyramid
        max_shape = pyr[-1][pool_number].shape
        # print("max_shape",max_shape)
        tss = pyr[0][0].tss
        fig = plt.figure()
        # ax=fig.add_subplot(1,1,1,projection="3d")
        nots = len(pyr)
        print("nots=", nots)
        for i in range(nots):
            rectangles = pyr[i]
            rect = rectangles[pool_number]
            fig.clf()
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            rect.plot_bins(ax, max_shape)
            fig.savefig(trunk + "{first_arg:04d}".format(first_arg=i) + ".pdf")
