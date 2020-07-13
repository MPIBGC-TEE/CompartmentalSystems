import matplotlib.patches as mpatches
import numpy as np


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

    def plot(self, ax):
        ax.add_patch(
            mpatches.Circle(
                (self.x, self.y),
                self.size, 
                alpha=self.pool_alpha,
                color=self.pool_color
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

    def plot_input_flux(self, ax, color):
        z1 = self.x-0.5 + (self.y-0.5)*1j
        arg1 = np.angle(z1) - np.pi/6
        
        z1 = z1 + np.exp(1j*arg1) * self.size
        x1 = 0.5+z1.real
        y1 = 0.5+z1.imag
        
        z2 = z1 + np.exp(1j * arg1) * self.size * 1.0
        
        x2 = 0.5+z2.real
        y2 = 0.5+z2.imag
        
        ax.add_patch(
            mpatches.FancyArrowPatch(
                (x2,y2),
                (x1,y1), 
                connectionstyle=self.connectionstyle,
                arrowstyle=self.arrowstyle, 
                mutation_scale=self.mutation_scale,
                alpha=self.pipe_alpha, 
                color=color
            )
        )
        
    def plot_output_flux(self, ax, color):
        z1 = self.x-0.5 + (self.y-0.5)*1j
        arg1 = np.angle(z1) + np.pi/6
        
        z1 = z1 + np.exp(1j*arg1) * self.size
        x1 = 0.5+z1.real
        y1 = 0.5+z1.imag
        
        z2 = z1 + np.exp(1j * arg1) * self.size *1.0
        
        x2 = 0.5+z2.real
        y2 = 0.5+z2.imag
        
        ax.add_patch(
            mpatches.FancyArrowPatch(
                (x1,y1),
                (x2,y2), 
                arrowstyle=self.arrowstyle,
                connectionstyle=self.connectionstyle,
                mutation_scale=self.mutation_scale,
                alpha=self.pipe_alpha, 
                color=color
            )
        )

    def plot_internal_flux_to(self, ax, pool_to, color):
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

        ax.add_patch(
            mpatches.FancyArrowPatch(
                (x1,y1),
                (x2,y2), 
                connectionstyle=self.connectionstyle,
                arrowstyle=self.arrowstyle, 
                mutation_scale=self.mutation_scale,
                alpha=self.pipe_alpha,
                color=color
            )
        )
        

