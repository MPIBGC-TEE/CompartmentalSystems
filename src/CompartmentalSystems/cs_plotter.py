import matplotlib.patches as mpatches
import numpy as np
from sympy import Symbol, latex


class Pool():
    def __init__(
            self,
            x,
            y,
            size,
            pool_color,
            pool_alpha,
            pipe_alpha,
            connectionstyle,
            arrowstyle,
            mutation_scale
        ):
        self.x = x
        self.y = y
        self.size = size
        self.pool_color = pool_color
        self.pool_alpha = pool_alpha
        self.pipe_alpha = pipe_alpha
        self.connectionstyle = connectionstyle
        self.arrowstyle = arrowstyle
        self.mutation_scale = mutation_scale

    def plot(self, ax, black_and_white):
        if not black_and_white:
            ax.add_patch(
                mpatches.Circle(
                    (self.x, self.y),
                    self.size,
                    alpha=self.pool_alpha,
                    color=self.pool_color
                )
            )
        else:
            ax.add_patch(
                mpatches.Circle(
                    (self.x, self.y),
                    self.size,
                    edgecolor="black",
                    facecolor="white",
                    alpha=1.0
                )
            )


    def plot_name(self, ax, name, fontsize):
        ax.text(
            self.x,
            self.y,
            name,
            fontsize = fontsize,
            horizontalalignment='center',
            verticalalignment='center'
        )

    def plot_input_flux(self, ax, color, black_and_white):
        z1 = self.x-0.5 + (self.y-0.5)*1j
        arg1 = np.angle(z1) - np.pi/6

        z1 = z1 + np.exp(1j*arg1) * self.size
        x1 = 0.5+z1.real
        y1 = 0.5+z1.imag

        z2 = z1 + np.exp(1j * arg1) * self.size * 1.0

        x2 = 0.5+z2.real
        y2 = 0.5+z2.imag

        if not black_and_white:
            ax.add_patch(
                mpatches.FancyArrowPatch(
                    (x2,y2),
                    (x1,y1),
                    connectionstyle=self.connectionstyle,
                    arrowstyle=self.arrowstyle,
                    mutation_scale=0.5*self.mutation_scale,
                    alpha=self.pipe_alpha,
                    color=color
                )
            )
        else:
            ax.add_patch(
                mpatches.FancyArrowPatch(
                    (x2,y2),
                    (x1,y1),
                    connectionstyle=self.connectionstyle,
                    arrowstyle="simple",
                    mutation_scale=0.5*self.mutation_scale,
                    alpha=1.0,
                    fill=False,
                    color="black"
                )
            )


    def plot_output_flux(self, ax, color, black_and_white):
        z1 = self.x-0.5 + (self.y-0.5)*1j
        arg1 = np.angle(z1) + np.pi/6

        z1 = z1 + np.exp(1j*arg1) * self.size
        x1 = 0.5+z1.real
        y1 = 0.5+z1.imag

        z2 = z1 + np.exp(1j * arg1) * self.size *1.0

        x2 = 0.5+z2.real
        y2 = 0.5+z2.imag

        if not black_and_white:
            ax.add_patch(
                mpatches.FancyArrowPatch(
                    (x1,y1),
                    (x2,y2),
                    arrowstyle=self.arrowstyle,
                    connectionstyle=self.connectionstyle,
                    mutation_scale=0.5*self.mutation_scale,
                    alpha=self.pipe_alpha,
                    color=color
                )
            )
        else:
            ax.add_patch(
                mpatches.FancyArrowPatch(
                    (x1,y1),
                    (x2,y2),
                    arrowstyle="simple",
                    connectionstyle=self.connectionstyle,
                    mutation_scale=0.5*self.mutation_scale,
                    alpha=1.0,
                    fill=False,
                    color="black"
                )
            )


    def plot_internal_flux_to(self, ax, pool_to, color, black_and_white):
        r=self.size
        z1 = (self.x-0.5) + (self.y-0.5) * 1j
        z2 = (pool_to.x-0.5) + (pool_to.y-0.5) * 1j

        arg1 = np.angle(z2-z1) - np.pi/20
        z1 = z1+np.exp(1j*arg1)*r

        arg2 = np.angle(z1-z2)  + np.pi/20
        z2 = z2+np.exp(1j*arg2)*r

        x1 = 0.5+z1.real
        y1 = 0.5+z1.imag

        x2 = 0.5+z2.real
        y2 = 0.5+z2.imag

        if not black_and_white:
            ax.add_patch(
                mpatches.FancyArrowPatch(
                    (x1,y1),
                    (x2,y2),
                    connectionstyle=self.connectionstyle,
                    arrowstyle=self.arrowstyle,
                    mutation_scale=0.5*self.mutation_scale,
                    alpha=self.pipe_alpha,
                    color=color
                )
            )
        else:
            ax.add_patch(
                mpatches.FancyArrowPatch(
                    (x1,y1),
                    (x2,y2),
                    connectionstyle=self.connectionstyle,
                    arrowstyle="simple",
                    mutation_scale=0.5*self.mutation_scale,
                    alpha=1.0,
                    fill=False,
                    color="black"
                )
            )

class CSPlotter(): 
    def __init__(
            self,
            state_vector, 
            inputs, 
            outputs,  
            internal_fluxes, 
            pipe_colors = {
                'linear': 'blue',
                'nonlinear': 'green',
                'no state dependence': 'red',
                'undetermined': 'grey'
            },
            visible_pool_names = True, 
            pool_size_scale_factor  = 1, 
            pool_color = 'blue',
            pool_alpha = 0.3,
            pipe_alpha = 0.5,
            connectionstyle = 'arc3, rad=0.1',
            arrowstyle = 'simple',
            mutation_scale = 50,
            fontsize = 24
        ):
        self.state_vector = state_vector
        self.input_fluxes= inputs
        self.output_fluxes = outputs
        self.internal_fluxes = internal_fluxes
        self.visible_pool_names= visible_pool_names
        self.pool_size_scale_factor = pool_size_scale_factor
        self.pipe_colors = pipe_colors
        self.pool_color = pool_color
        self.pool_alpha = pool_alpha
        self.pipe_alpha = pipe_alpha
        self.connectionstyle = connectionstyle
        self.arrowstyle = arrowstyle
        self.mutation_scale = mutation_scale
        self.fontsize = fontsize

    def plot_pools_and_fluxes(self, ax, black_and_white=False):
        nr_pools = len(self.state_vector)
        inputs = self.input_fluxes
        outputs =  self.output_fluxes
        internal_fluxes = self.internal_fluxes
        base_r = 0.1 + (0.5-0.1)/10*nr_pools
        base_r = min(base_r, 0.4)

        if nr_pools > 1:
            r = base_r * (1-np.exp(1j*2*np.pi/nr_pools))
            r = abs(r) / 2 * 0.6
            r = min(r, (0.5-base_r)*0.5)
        else:
            r = base_r * 0.5

        r = abs(r)
        r = r * self.pool_size_scale_factor
        r = max(r, 0.05)

        pools = []
        for i in range(nr_pools):
            z = base_r * np.exp(i*2*np.pi/nr_pools*1j)
            x = 0.5 - z.real
            y = 0.5 + z.imag

            pool = Pool(
                x,
                y,
                r,
                self.pool_color,
                self.pool_alpha,
                self.pipe_alpha,
                self.connectionstyle,
                self.arrowstyle,
                self.mutation_scale
            )
            pool.plot(ax, black_and_white)

            if self.visible_pool_names:
                pool_name = Symbol(str(self.state_vector[i]))
                pool.plot_name(ax, "$"+latex(pool_name)+"$", self.fontsize)

            # plot influx
            if i in inputs.keys():
                pool.plot_input_flux(
                    ax,
                    self.pipe_colors[inputs[i]],
                    black_and_white
                )

            # plot outflux
            if i in outputs.keys():
                pool.plot_output_flux(
                    ax,
                    self.pipe_colors[outputs[i]],
                    black_and_white
                )

            pools.append(pool)
        
        # plot internal fluxes
        for key in internal_fluxes.keys():
            i, j = key
            pools[i].plot_internal_flux_to(
                ax,
                pools[j],
                self.pipe_colors[internal_fluxes[key]],
                black_and_white
            )
        

    def legend(self, ax):
        legend_descs = []
        legend_colors = []
        for desc, col in self.pipe_colors.items():
            legend_descs.append(desc)
            legend_colors.append(
                mpatches.FancyArrowPatch(
                    (0,0),
                    (1,1),
                    connectionstyle=self.connectionstyle,
                    arrowstyle=self.arrowstyle,
                    mutation_scale=self.mutation_scale,
                    alpha=self.pipe_alpha,
                    color=col
                )
            )

        ax.legend(legend_colors, legend_descs, loc='upper center',
                    bbox_to_anchor=(0.5, 1.1), ncol = 3)

