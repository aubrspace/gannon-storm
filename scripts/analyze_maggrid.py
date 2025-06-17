#!/usr/bin/env python3
"""Analyze and plot magnetometer data from the Gannon storm
"""
import os,sys,glob,time
import numpy as np
from numpy import sin,cos,deg2rad,rad2deg,pi
import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt
from cmcrameri import cm
#
from global_energetics.extract.magnetometer import (read_MGL,loadmagnetometers)
from global_energetics.analysis.plot_tools import (pyplotsetup,
                                                   bin_and_describe,
                                                   extended_fill_between,
                                                   general_plot_settings)
#
from supermag_api import (SuperMAGGetInventory,SuperMAGGetData,
                          SuperMAGGetIndices)

def add_station_compare(ax:plt.Axes,swmf:pd.DataFrame,station:str) ->plt.Axes:
    # Call supermag to get the obs data
    #   NOTE pass by reference here, sm is a global dict
    sm = proc_supermag(station,swmf.index[0],swmf.index[-1])
    swmf_single = swmf[swmf['IAGA']==station]
    # Draw both lines on this axis with some settings
    ax.plot(sm.index,sm['dBn'],label='dBn_sm',c='black')
    ax.plot(swmf_single.index,swmf_single['dBn'],label='dBn_swmf',c='magenta')
    ax.plot(sm.index,sm['dBe'],label='dBe_sm',c='grey')
    ax.plot(swmf_single.index,swmf_single['dBe'],label='dBe_swmf',c='cyan')
    ax.plot(sm.index,sm['dBd'],label='dBd_sm',c='saddlebrown')
    ax.plot(swmf_single.index,swmf_single['dBd'],label='dBd_swmf',c='lime')
    return ax

def plot_station_comparisons(swmf:pd.DataFrame,path:str) -> None:
    #stations = ['LRV','HOV','LER','KAR']
    #stations = ['AIA','ORC','STJ']
    stations = ['PIN','MGD','SBL','SHU']
    # Figure
    fig,axes = plt.subplots(len(stations),figsize=[24,8*len(stations)])
    #from IPython import embed; embed()
    for i,(station,ax) in enumerate(zip(stations,axes)):
        print(f"Plotting {station}")
        # Plot
        ax = add_station_compare(ax,swmf,station)
        # Decorate
        general_plot_settings(ax,do_xlabel=i==(len(stations)-1),legend=False,
                              ylabel=f'{station}'+r' dB $\left[nT\right]$',
                              xlim=[swmf.index[0],swmf.index[-1]],
                              timedelta=False)
    axes[0].legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                   ncol=3, fancybox=True, shadow=True)
    # Save
    fig.tight_layout(pad=1)
    figurename = f"{path}/magnetometers.png"
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)

def proc_supermag(station:str,start:dt.datetime,
                                end:dt.datetime) -> pd.DataFrame:
    t0 = dt.datetime(1970,1,1)
    userid = 'aubr'
    duration = int((end-start).total_seconds())
    status,sm_data = SuperMAGGetData(userid,start.isoformat(),duration,
                                                                   '',station)
    # Adjust the columns to be easier to work with
    sm_data['time'] = [t0+dt.timedelta(seconds=t) for t in sm_data['tval']]
    sm_data.index = sm_data['time']
    sm_data.drop(columns=['tval','time'],inplace=True)
    sm_data['dBn'] = [v['nez'] for v in sm_data['N']]
    sm_data['dBe'] = [v['nez'] for v in sm_data['E']]
    sm_data['dBd'] = [v['nez'] for v in sm_data['Z']]
    return sm_data

def main() -> None:
    global TINIT, TIMPACT, TMIN, TEND, sm
    TINIT = dt.datetime(2024,5,10,12,0)
    TIMPACT = dt.datetime(2024,5,10,17)
    TMIN  = dt.datetime(2024,5,11,1,30)
    TEND  = dt.datetime(2024,5,11,18)
    inBase = os.path.realpath('..')+'/'
    inLogs = os.path.join(inBase,'data/logs/')
    inSats = os.path.join(inBase,'data/sat/')
    inAnalysis = os.path.join(inBase,'data/analysis/')
    outPath = os.path.join(inBase,'outputs/figures')
    unfiled = os.path.join(outPath,'unfiled')

    for path in [outPath,unfiled]:
        os.makedirs(path,exist_ok=True)
    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='print'))

    #dataset = {}
    ## Analysis Data
    magfile = "../data/large/GM/IO2/magnetometers_e20240510-130000.mag"
    swmf = loadmagnetometers(magfile)
    #grid = read_MGL("../data/large/GM/IO2/")

    ## Supermag Data - will be downloaded as needed, saved into global var
    sm = {}

    ## Plot
    plot_station_comparisons(swmf,unfiled)
    #from IPython import embed; embed()


if __name__ == "__main__":
    main()
