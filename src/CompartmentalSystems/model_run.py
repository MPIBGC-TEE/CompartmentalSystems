import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod, abstractproperty


def plot_attributes(mrs, file_name):
    colors = ['red', 'blue', 'orange', 'green', 'yellow', 'black']
    labels = ['mr_ref', 'mr_1', 'mr_2', 'mr_3', 'mr_4', 'mr_5']
    markersizes = [8, 6, 4, 2, 1, 0.5]
    lc = len(colors)
    if len(mrs) > lc:
        raise(Exception("only "+str(lc)+" different modelruns supported."))
    else:
        meths = [
            'solve',
            'acc_net_external_input_vector',
            'acc_net_external_output_vector'
        ]
        nr_pools = mrs[0].nr_pools
        fig, axs = plt.subplots(
            nrows=len(meths),
            ncols=nr_pools,
            gridspec_kw={'hspace': 0.3, 'wspace': 0.1},
            figsize=(8.27, 11.69)
        )

        for j, meth in enumerate(meths):
            for i in range(nr_pools):
                ax = axs[j, i]
                ax.set_title(meth+", " + str(i))
                for k, mr in enumerate(mrs):
                    y = getattr(mr, meth)()[:, i],
                    ax.plot(
                        mr.times[:len(y)],
                        y,
                        '*',
                        color=colors[k],
                        label=labels[k],
                        markersize=markersizes[k]
                    )
                ax.legend()

    fig.savefig(file_name, tight_layout=True)


def plot_stocks_and_fluxes(mrs, file_name, labels=None):
    colors = ['red', 'blue', 'orange', 'green', 'yellow', 'black']
    if labels is None:
        labels = ['mr_ref', 'mr_1', 'mr_2', 'mr_3', 'mr_4', 'mr_5']
    markersizes = [8, 6, 4, 2, 1, 0.5]
    lc = len(colors)
    if len(mrs) > lc:
        raise(Exception("only "+str(lc)+" different modelruns supported."))
    else:
        nr_pools = mrs[0].nr_pools
        fig, axs = plt.subplots(
            nrows=nr_pools+1,
            ncols=nr_pools+1,
            gridspec_kw={'hspace': 0.3, 'wspace': 0.1},
            figsize=(11.69, 11.69)
        )

        # solutions
        meth = 'solve'
        for i in range(nr_pools):
            ax = axs[i, i+1]
            ax.set_title(meth + ", " + str(i))
            for k, mr in enumerate(mrs):
                y = getattr(mr, meth)()[:, i]
                ax.plot(
                    mr.times[:len(y)],
                    y,
                    '*',
                    color=colors[k],
                    label=labels[k],
                    markersize=markersizes[k]
                )
                ax.legend()

        def f(X, Y): return X / Y[:len(X)]
        for symb, net_or_gross in zip(["o", "*-"], ["gross", "net"]):
            # influxes
            tit = 'acc external input vector'
            meth = 'acc_'+net_or_gross+'_external_input_vector'
            for i in range(nr_pools):
                ax = axs[i, 0]
                ax.set_title(tit + ", " + str(i))
                for k, mr in enumerate(mrs):
                    if hasattr(mr, meth):
                        y = f(getattr(mr, meth)()[:, i], mr.dts)
                        ax.plot(
                            mr.times[:len(y)],
                            y,
                            symb,
                            color=colors[k],
                            label=labels[k]+'_'+net_or_gross,
                            markersize=markersizes[k]
                        )
                ax.legend()

            # outfluxes
            tit = 'acc external output vector'
            meth = 'acc_'+net_or_gross+'_external_output_vector'
            for j in range(nr_pools):
                ax = axs[-1, j+1]
                ax.set_title(tit + ", " + str(j))
                for k, mr in enumerate(mrs):
                    if hasattr(mr, meth):
                        y = f(getattr(mr, meth)()[:, j], mr.dts)
                        ax.plot(
                            mr.times[:len(y)],
                            y,
                            symb,
                            color=colors[k],
                            label=labels[k]+'_'+net_or_gross,
                            markersize=markersizes[k]
                        )
                ax.legend()

            # internal fluxes
            meth = 'acc_'+net_or_gross+'_internal_flux_matrix'
            for i in range(nr_pools):
                for j in range(nr_pools):
                    if i != j:
                        ax = axs[i, j+1]
                        ax.set_title(
                            'F({0},{1}) = acc flux from {1} to {0}'
                            .format(i, j)
                        )
                        for k, mr in enumerate(mrs):
                            if hasattr(mr, meth):
                                y = f(getattr(mr, meth)()[:, i, j], mr.dts)
                                ax.plot(
                                    mr.times[:len(y)],
                                    y,
                                    symb,
                                    color=colors[k],
                                    label=labels[k]+'_'+net_or_gross,
                                    markersize=markersizes[k]
                                )
                        ax.legend()

        axs[nr_pools, 0].set_visible(False)
        fig.savefig(file_name, tight_layout=True)


class ModelRun(metaclass=ABCMeta):
    # abstractmehtods HAVE to be overloaded in the subclasses
    # the decorator should only be used inside a class definition

    @abstractproperty
    def nr_pools(self):
        pass

    @abstractproperty
    def dts(self):
        pass

    @abstractmethod
    def solve(self, alternative_start_values=None):
        pass

    @abstractmethod
    def acc_gross_external_input_vector(self):
        pass

    @abstractmethod
    def acc_gross_internal_flux_matrix(self):
        pass

    @abstractmethod
    def acc_gross_external_output_vector(self):
        pass

    @abstractmethod
    def acc_net_external_input_vector(self):
        pass

    @abstractmethod
    def acc_net_internal_flux_matrix(self):
        pass

    @abstractmethod
    def acc_net_external_output_vector(self):
        pass
