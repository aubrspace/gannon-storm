#!/usr/bin/env python3
"""Analysis and plots for the Gannon storm GRL paper comparisons with DMSP
"""
import os,sys,glob,time
import numpy as np
from numba import jit,prange
import datetime as dt
from matplotlib import pyplot as plt
from tqdm import tqdm
#
from global_energetics.analysis.plot_tools import (pyplotsetup,
                                                   general_plot_settings)

def draw_dmsp_north_tseries(axis:plt.Axes,dmsp:dict,**kwargs:dict) -> None:
    axis.plot(dmsp['F16_N']['time'],dmsp['F16_N']['cpcp_kV'],label='dmspF16',
              color='red')
    axis.plot(dmsp['F17_N']['time'],dmsp['F17_N']['cpcp_kV'],label='dmspF17',
              color='blue')
    axis.plot(dmsp['F18_N']['time'],dmsp['F18_N']['cpcp_kV'],label='dmspF18',
              color='black')
    axis.scatter(dmsp['F16_N']['time'],dmsp['F16_N']['cpcp_kV'],label='_F16_N',
              color='red',s=150,marker='X')
    axis.scatter(dmsp['F17_N']['time'],dmsp['F17_N']['cpcp_kV'],label='_F17_N',
              color='blue',s=150,marker='X')
    axis.scatter(dmsp['F18_N']['time'],dmsp['F18_N']['cpcp_kV'],label='_F18_N',
              color='black',s=150,marker='X')
    return

def draw_dmsp_south_tseries(axis:plt.Axes,dmsp:dict,**kwargs:dict) -> None:
    axis.plot(dmsp['F16_S']['time'],dmsp['F16_S']['cpcp_kV'],label='dmspF16',
              color='red')
    axis.plot(dmsp['F17_S']['time'],dmsp['F17_S']['cpcp_kV'],label='dmspF17',
              color='blue')
    axis.plot(dmsp['F18_S']['time'],dmsp['F18_S']['cpcp_kV'],label='dmspF18',
              color='black')
    axis.scatter(dmsp['F16_S']['time'],dmsp['F16_S']['cpcp_kV'],label='_F16_S',
              color='red',s=150,marker='X')
    axis.scatter(dmsp['F17_S']['time'],dmsp['F17_S']['cpcp_kV'],label='_F17_S',
              color='blue',s=150,marker='X')
    axis.scatter(dmsp['F18_S']['time'],dmsp['F18_S']['cpcp_kV'],label='_F18_S',
              color='black',s=150,marker='X')
    return

def draw_swmf_north_tseries(axis:plt.Axes,dmsp:dict,swmf:dict) -> None:
    cpcp = [p.max()-p.min() for p in swmf['PHI [kV]']]
    axis.fill_between(swmf['time'],cpcp,label='swmf_whole',fc='lightgrey')
    axis.plot(dmsp['F16_N']['time'],dmsp['F16_N']['ie_cpcp'],
                 label='swmfF16',color='orange',ls='--')
    axis.plot(dmsp['F17_N']['time'],dmsp['F17_N']['ie_cpcp'],
                 label='swmfF17',color='purple',ls='--')
    axis.plot(dmsp['F18_N']['time'],dmsp['F18_N']['ie_cpcp'],
                 label='swmfF18',color='dimgrey',ls='--')
    axis.scatter(dmsp['F16_N']['time'],dmsp['F16_N']['ie_cpcp'],
                 label='_swmfF16',color='orange',s=150,marker='o')
    axis.scatter(dmsp['F17_N']['time'],dmsp['F17_N']['ie_cpcp'],
                 label='_swmfF17',color='purple',s=150,marker='o')
    axis.scatter(dmsp['F18_N']['time'],dmsp['F18_N']['ie_cpcp'],
                 label='_swmfF18',color='dimgrey',s=150,marker='o')
    return

def draw_swmf_south_tseries(axis:plt.Axes,dmsp:dict,swmf:dict) -> None:
    cpcp = [p.max()-p.min() for p in swmf['PHI [kV]']]
    axis.fill_between(swmf['time'],cpcp,label='swmf_whole',fc='lightgrey')
    axis.plot(dmsp['F16_S']['time'],dmsp['F16_S']['ie_cpcp'],
                 label='swmfF16',color='orange',ls='--')
    axis.plot(dmsp['F17_S']['time'],dmsp['F17_S']['ie_cpcp'],
                 label='swmfF17',color='purple',ls='--')
    axis.plot(dmsp['F18_S']['time'],dmsp['F18_S']['ie_cpcp'],
                 label='swmfF18',color='dimgrey',ls='--')
    axis.scatter(dmsp['F16_S']['time'],dmsp['F16_S']['ie_cpcp'],
                 label='_swmfF16',color='orange',s=150,marker='o')
    axis.scatter(dmsp['F17_S']['time'],dmsp['F17_S']['ie_cpcp'],
                 label='_swmfF17',color='purple',s=150,marker='o')
    axis.scatter(dmsp['F18_S']['time'],dmsp['F18_S']['ie_cpcp'],
                 label='_swmfF18',color='dimgrey',s=150,marker='o')
    return

def plot_timeseries(ie:dict,dmsp:dict,path:str) -> None:
    # Figure
    fig,[ax1,ax2] = plt.subplots(2,1,figsize=[24,24],sharex=True)
    # Plot
    draw_swmf_north_tseries(ax1,dmsp,ie['N'])
    draw_swmf_south_tseries(ax2,dmsp,ie['S'])
    draw_dmsp_north_tseries(ax1,dmsp)
    draw_dmsp_south_tseries(ax2,dmsp)
    # Decorate
    for i,axis in enumerate([ax1,ax2]):
        general_plot_settings(axis,do_xlabel=i==1,legend=False,timedelta=False,
                              ylabel=r'CPCP $\left[kV\right]$',
                              xlim=[ie['N']['time'][0],
                                    ie['N']['time'][-1]])
    ax1.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                ncol=4, fancybox=True, shadow=True)
    ax1.text(0.9,0.9,"North",transform=ax1.transAxes,fontsize=36,
             horizontalalignment='right')
    ax2.text(0.9,0.9,"South",transform=ax2.transAxes,fontsize=36,
             horizontalalignment='right')
    fig.tight_layout()
    # Save
    figurename = f"{path}/tseries_all.png"
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)
##############################################################################
def draw_sat_cross(axis:plt.Axes,satellite:dict,
                  start:dt.datetime,end:dt.datetime) -> plt.scatter:
    interv = (satellite['time']>start) & (satellite['time']<end)
    sc = axis.scatter(satellite['mlt'][interv],90-satellite['mlat'][interv],
                      c=satellite['Ex'][interv])
    return sc

def plot_iono_projection(ie:dict,dmsp:dict,path:str) -> None:
    # Figure
    fig,axis = plt.subplots(1,1,figsize=[12,12],
                            subplot_kw={'projection':'polar'})
    sc = draw_sat_cross(axis,dmsp['F16_N'],dt.datetime(2024,5,10,17,0),
                                           dt.datetime(2024,5,11,17,0))
    #cb = fig.colorbar(sc,orientation='vertical',cax=axis)
    #cb.set_label(r"CPCP $\left[kV\right]$",fontsize=36)
    plt.show()
##############################################################################

@jit(nopython=True, parallel=True)
def dumb_2D_interp(x:np.ndarray,y:np.ndarray,z:np.ndarray,
           target_xs:np.ndarray,target_ys:np.ndarray,
                                                doReverse:bool) -> np.ndarray:
    """ x,y,z- three dimensional (t,X,Y)
    """
    z_target = np.zeros(z.shape[0])
    for it,zlocal in enumerate(z):
        target_x = target_xs[it]
        target_y = target_ys[it]
        # Interpolate in X
        z_interpX = np.zeros(z.shape[1])
        for iy in prange(0,z.shape[1]):
            if doReverse:
                z_interpX[iy] = np.interp(target_x,x[0,iy,:][::-1],
                                                   zlocal[iy,:][::-1])
            else:
                z_interpX[iy] = np.interp(target_x,x[0,iy,:],zlocal[iy,:])
        # Interpolate in Y
        z_target[it] = np.interp(target_y,y[0,:,0],z_interpX)
    return z_target


def extract_ie_sample(indices:np.ndarray,satellite:dict,ie:dict)-> np.ndarray:
    # Get time into interpolateable format
    t_sat = np.array([(t-T0).total_seconds()
                                         for t in satellite['time'][indices]])
    t_ie = np.array([(t-T0).total_seconds() for t in ie['time']])

    # Reshape some data for later
    theta = (90-ie['Theta [deg]']).reshape(len(t_ie),181,91)
    psi   = ie['Psi [deg]'].reshape(len(t_ie),181,91)
    ie_pot= ie['PHI [kV]']

    # Reverse bc np.interp expects increasing values
    if theta[0,0,0]>theta[0,0,-1]:
        doReverse = True

    # Initialize output array
    #pot_target = np.zeros(len(t_sat))

    # Interpolate in time
    ie_pot_t_interp = np.array([np.interp(t_sat,t_ie,ie_pot[:,i])
                              for i in range(0,ie_pot.shape[-1])]).transpose()
    ie_pot_t_interp = ie_pot_t_interp.reshape(len(t_sat),181,91)

    pot_target = dumb_2D_interp(theta,psi,ie_pot_t_interp,
                                satellite['mlat'][indices],
                                satellite['mlon'][indices],doReverse)
    '''
    # Dumb loop in time, now interpolate twice in space
    for it,pot in enumerate(ie_pot_t_interp.reshape(len(t_sat),181,91)):
        target_lat = satellite['mlat'][it]
        target_lon = (satellite['mlt'][it]/12*180+180)%360
        # Interpolate in lat
        pot_lat_interp = np.array(
                        [np.interp(target_lat,theta[0,ilon,:],pot[ilon,:])
                                                for ilon in range(0,181)])
        # Interpolate in lon
        pot_target[it] = np.interp(target_lon,psi[0,:,0],pot_lat_interp)
    '''
    return pot_target

def main() -> None:
    plt.rcParams.update(pyplotsetup(mode='print'))
    # Load in .npz files
    ie = {}
    ie['N'] = dict(np.load("../data/large/IE/ionosphere/compiled_N.npz",
                           allow_pickle=True))
    ie['S'] = dict(np.load("../data/large/IE/ionosphere/compiled_S.npz",
                           allow_pickle=True))

    dmsp = {}
    dmsp['F16_N'] = dict(np.load("../data/dmsp/compiled_F16_N.npz",
                           allow_pickle=True))
    dmsp['F17_N'] = dict(np.load("../data/dmsp/compiled_F17_N.npz",
                           allow_pickle=True))
    dmsp['F18_N'] = dict(np.load("../data/dmsp/compiled_F18_N.npz",
                           allow_pickle=True))
    dmsp['F16_S'] = dict(np.load("../data/dmsp/compiled_F16_S.npz",
                           allow_pickle=True))
    dmsp['F17_S'] = dict(np.load("../data/dmsp/compiled_F17_S.npz",
                           allow_pickle=True))
    dmsp['F18_S'] = dict(np.load("../data/dmsp/compiled_F18_S.npz",
                           allow_pickle=True))

    #for sat in dmsp:
    if False:
        # Initialize arrays inside dmsp dict to hold the extraction result
        dmsp[sat]['ie_pot'] = np.zeros(dmsp[sat]['pot'].shape)
        dmsp[sat]['ie_cpcp'] = np.zeros(dmsp[sat]['cpcp_kV'].shape)
        dmsp[sat]['mlon'] = np.array([(mlt/12*180+180)%360
                                                for mlt in dmsp[sat]['mlt']])
        # Find the range of indices for a single cpcp calculation
        for pass_id in tqdm(np.unique(dmsp[sat]['pass_id'])):
            ipass = dmsp[sat]['pass_id']==pass_id
            cpcp_values = np.unique(dmsp[sat]['cpcp_kV'][ipass])
            for cpcp in [v for v in cpcp_values if not np.isnan(v)]:
                icpcp = dmsp[sat]['cpcp_kV'][ipass] == cpcp
                isample = np.where(ipass)[0][icpcp]
                # Do the extraction for just this one interval
                if 'N' in sat:
                    ie_pot =extract_ie_sample(isample,dmsp[sat],ie['N'])
                elif 'S' in sat:
                    ie_pot =extract_ie_sample(isample,dmsp[sat],ie['S'])
                # Store the data in our initialized arrays
                dmsp[sat]['ie_pot'][isample] = ie_pot
                dmsp[sat]['ie_cpcp'][isample] = ie_pot.max()-ie_pot.min()
        np.savez_compressed(f"../data/dmsp/compiled_{sat}.npz",**dmsp[sat])
        print(f'\033[92m Created\033[00m "../data/dmsp/compiled_{sat}.npz"')

    #plot_iono_projection(ie,dmsp,"../outputs/dmsp")
    #plot_timeseries(ie,dmsp,"../outputs/figures/dmsp")

    # For each satellite
    #   Plot just the crossings in N hemi SM coordinated w/ Ex color
    #   Downsample to just the crossing times
    #   From each crosing
    #       extract IE along the crossing
    #       integrate E along each crossing
    #       adjustment for altitude???
    return

if __name__ == "__main__":
    start_time = time.time()

    global T0
    T0 = dt.datetime(2024,5,10,13,0)

    main()

    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
