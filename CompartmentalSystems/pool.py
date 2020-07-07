import matplotlib.patches as mpatches
import numpy as np


class Pool():
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size

    def plot(self, ax, pool_color, pool_alpha):
        ax.add_patch(
            mpatches.Circle(
                (self.x, self.y),
                self.size, 
                alpha=pool_alpha,
                color=pool_color
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

    def plot_input_flux(self, ax, color, alpha, arrowstyle, mutation_scale):
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
                connectionstyle='arc3, rad=0.1', 
                arrowstyle=arrowstyle, 
                mutation_scale=mutation_scale,
                alpha=alpha, 
                color=color
            )
        )
        
    def plot_output_flux(self, ax, color, alpha, arrowstyle, mutation_scale):
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
                arrowstyle=arrowstyle,
                connectionstyle='arc3, rad=0.1', 
                mutation_scale=mutation_scale,
                alpha=alpha, 
                color=color
            )
        )




