# vim: set ff=unix expandtab ts=4 sw=4:
from copy import deepcopy
from string import Template
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .plot_helpers import cc,SinglePlotFigureHandler, content_facecolors, deathrate_facecolors, loss_facecolors, gain_facecolors
from .TstBin import TstBin
from .TsTpMassField import TsTpMassField

def plot_external_gains(external_input_numbers,tss,nTs,nTp,max_z):
    for key,number in external_input_numbers.items():
        
        tuplestr='{first_arg:01d}'.format(first_arg=key)
        with SinglePlotFigureHandler("external_gains_pool"+tuplestr+".pdf") as ax:
            ax.set_title(Template("external gains for pool  ${key}").substitute(key=key))
            f=TsTpMassField(np.zeros((nTs,nTp)),tss)
            f[0,0]=number
            f.plot_bins(ax,facecolors=gain_facecolors)
            ax.set_zlim3d(0,max_z)
        


def plot_figure_per_pipe(rectangles,max_Ts,max_Tp,max_z,title_stem,facecolors):
    # this could be a method of TsTpMassFieldsPerPipe
    # but maybe it is a bit special...
    for key,f in rectangles.items():
        tupelstr='{first_arg:01d}_{second_arg:01d}'.format(first_arg=key[0],second_arg=key[1])
        with SinglePlotFigureHandler(title_stem.replace(" ","_")+tupelstr+".pdf") as ax:
            ax.set_title(title_stem +str(key))
            f.plot_bins(ax,facecolors=facecolors)
            ax.set_xlim3d(0,max_Ts)
            ax.set_ylim3d(0,max_Tp)
            ax.set_zlim3d(0,max_z)

def plot_figure_per_pool(rectangles,max_Ts,max_Tp,max_z,title_stem,facecolors):
    # this could be a method of TsTpMassFieldsPerPool
    # but maybe it is a bit special...
    for key,f in rectangles.items():
        with SinglePlotFigureHandler(title_stem.replace(" ","_")+'{first_arg:01d}'.format(first_arg=key)+".pdf") as ax:
            ax.set_title(title_stem +str(key))
            f.plot_bins(ax,facecolors=facecolors)
            ax.set_xlim3d(0,max_Ts)
            ax.set_ylim3d(0,max_Tp)
            ax.set_zlim3d(0,max_z)

def plot_contents_and_gains(contents,internal_gains,external_gains,max_Ts,max_Tp,max_z):
    for content_pool_key,content in contents.items():
        with SinglePlotFigureHandler("contents_and_gains_pool_"+'{first_arg:01d}'.format(first_arg=content_pool_key)+".pdf") as ax:
            ax.set_title(Template("contents of pool ${key} and all gains").substitute(key=content_pool_key))
            offset=content.arr[:,0] # since gains can be at most one dimensional
            for receiver,gain in internal_gains.items():
                if receiver==content_pool_key:
                    gain.plot_bins(ax,0,facecolors=gain_facecolors,offset_field=offset)
                    offset+=gain.arr
            for receiver,gain in external_gains.items():
                if receiver==content_pool_key:
                    b=TstBin(content.tss,0,0,gain,facecolors=gain_facecolors)
                    b.plot(ax)
                    offset[0]+=gain
            content.plot_bins(ax,facecolors=content_facecolors)
            ax.set_xlim3d(0,max_Ts)
            ax.set_ylim3d(0,max_Tp)
            ax.set_zlim3d(0,max_z)

def plot_contents_and_losses(remainders,internal_losses,external_losses,max_Ts,max_Tp,max_z):
    for content_pool_key,this_pool_remainder in remainders.items():

        with SinglePlotFigureHandler("contents_and_losses_pool_"+'{first_arg:01d}'.format(first_arg=content_pool_key)+".pdf") as ax:
            ax.set_title(Template("contents of pool ${key} and all losses").substitute(key=content_pool_key))
            
            offset=this_pool_remainder.arr
            for pipe_key,loss in internal_losses.items():
                sender=pipe_key[0]
                if sender==content_pool_key:
                    loss.plot_bins(ax,facecolors=loss_facecolors,offset_field=offset)
                    offset+=loss.arr

            for pipe_key,loss in external_losses.items():
                sender=pipe_key
                print(loss.arr)
                if sender==content_pool_key:
                    loss.plot_bins(ax,facecolors=loss_facecolors,offset_field=offset)
                    offset+=loss.arr

            this_pool_remainder.plot_bins(ax,facecolors=content_facecolors)
            ax.set_xlim3d(0,max_Ts)
            ax.set_ylim3d(0,max_Tp)
            ax.set_zlim3d(0,max_z)

def plot_internal_gains(gains,max_Ts,max_Tp,max_z):
    # internal gains per pool
    for key,f in gains.items():
        
        tuplestr='{first_arg:01d}'.format(first_arg=key)
        with SinglePlotFigureHandler("internal_gains_pool"+tuplestr+".pdf") as ax:
            ax.set_title(Template("internal gains for pool  ${key}").substitute(key=key))
            #f.plot_bins(ax,0,t_min=0,t_max=3,facecolors=gain_facecolors)
            f.plot_bins(ax,0,facecolors=gain_facecolors)
            ax.set_xlim3d(0,max_Ts)
            ax.set_ylim3d(0,max_Tp)
            ax.set_zlim3d(0,max_z)


class TimeStepPlotter:
    def __init__(self,time_step):
        self.time_step=time_step
    def plot_pdfs(self):
        ts=self.time_step
        updated_content=ts.updated_content
        external_death_rate_fields=ts.external_death_rate_fields
        internal_death_rate_fields=ts.internal_death_rate_fields
        external_input_numbers=ts.external_input_numbers
        res=deepcopy(ts.rectangles)
        external_losses=res.external_losses(external_death_rate_fields)
        internal_losses=res.internal_losses(internal_death_rate_fields)
        gains=internal_losses.gains
        res=deepcopy(ts.rectangles)
        # first determine the plotting area (Tsmax,Tpmax,zmax)
        # so that all plots get the same size which looks nicer for a cartoon
        ts.rectangles.update(ts.updated_content)
        tss=ts.rectangles[0].tss
        nTs=max([c.number_of_Ts_entries    for c in ts.rectangles.values()])
        max_Ts=nTs*tss
        
        nTp=max([c.number_of_Tp_entries    for c in ts.rectangles.values()])
        max_Tp=nTp*tss
        
        max_z= max([c.arr.max() for c in ts.rectangles.values()])

        plot_figure_per_pool(res,max_Ts,max_Tp,max_z,"initial content pool ",content_facecolors)
        
        res.remove(external_losses)
        res.remove(internal_losses)
        remainder=deepcopy(res)
        plot_contents_and_losses(remainder,internal_losses,external_losses,max_Ts,max_Tp,max_z)
        plot_figure_per_pool(external_losses,max_Ts,max_Tp,max_z,"external losses pool ",loss_facecolors)
        plot_figure_per_pool(external_death_rate_fields,max_Ts,max_Tp,max_z,"external death rates pool ",deathrate_facecolors)
        plot_figure_per_pipe(internal_death_rate_fields,max_Ts,max_Tp,max_z,"internal death rates pipe ",deathrate_facecolors)
        
        plot_figure_per_pipe(internal_losses,max_Ts,max_Tp,max_z,"internal losses pipe ",loss_facecolors)
        plot_figure_per_pool(res,max_Ts,max_Tp,max_z,"content after removel of internal and external losses pool ",content_facecolors)
        res.shift()
        plot_figure_per_pool(res,max_Ts,max_Tp,max_z,"contents after shift",content_facecolors)

        plot_contents_and_gains(res,gains,external_input_numbers,max_Ts,max_Tp,max_z)
        res.receive(gains)
        res.receive_external(external_input_numbers)
        plot_figure_per_pool(res,max_Ts,max_Tp,max_z,"updated contents pool ",content_facecolors)
        # 
        plot_external_gains(external_input_numbers,res[0].tss,nTs,nTp,max_z)
        plot_internal_gains(gains,max_Ts,max_Tp,max_z)


            
