#!/usr/bin/env python3
"""Analyze and plot data from the Gannon storm
"""
import os,sys,glob,time
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from global_energetics.analysis.plot_tools import (pyplotsetup,general_plot_settings)
from global_energetics.analysis.proc_hdf import load_hdf_sort

def central_difference(f:np.ndarray) -> np.ndarray:
    dt = 60 #NOTE
    dfdt = np.zeros_like(f)
    # 5 point central diff, assume boundaries are simply df/dt = 0
    dfdt[2:-2] = (-f[4::]+8*f[3:-1]-8*f[1:-3]+f[0:-4])/(12*dt)
    return dfdt

def get_normal_flux(ien:dict,ies:dict) -> list[float,float]:
    # Set dipole parameters
    axis = [0,0,-1] # assumed SM, no tilt
    B0 = 31000 #nT
    flux = np.zeros([2,len(ien['time'])])
    for ihemi,ie in enumerate([ien,ies]):
        # Caclulate dipole field at each point
        X = ie['X [R]']*(1+300/6371)
        Y = ie['Y [R]']*(1+300/6371)
        Z = ie['Z [R]']*(1+300/6371)
        R = np.sqrt(X**2+Y**2+Z**2)
        M11 = 3*X**2-R**2
        M12 = 3*X*Y
        M13 = 3*X*Z
        M21 = M12
        M22 = 3*Y**2-R**2
        M23 = 3*Y*Z
        M31 = M13
        M32 = M23
        M33 = 3*Z**2-R**2
        Bx = B0/R**5*(M11*axis[0] + M12*axis[1] + M13*axis[2])
        By = B0/R**5*(M21*axis[0] + M22*axis[1] + M23*axis[2])
        Bz = B0/R**5*(M31*axis[0] + M32*axis[1] + M33*axis[2])
        Br = (Bx*X + By*Y + Bz*Z)/R
        # Calculate Area
        dtheta = 1*np.pi/180 #NOTE
        dpsi = 2*np.pi/180 #NOTE
        Area = R**2*np.sin(np.pi/180*ie['Theta [deg]'])*dtheta*dpsi
        # Use 1/B mapping variable to find open/closed boundary
        binv = ie['RT 1/B [1/T]']
        # Integrate over where the field should be open (1/B < 0)
        for itime in tqdm(range(0,X.shape[0])):
            #openflux = binv[itime,:]<-1e5
            openflux = binv[itime,:]<-1e7
            flux[ihemi,itime] = np.sum(Br[itime,:][openflux]*
                                     Area[itime,:][openflux]*6371**2/1000)
    return flux

def draw_cpcp_panel(ax:plt.Axes,
               ietimes:np.ndarray,
                  cpcp:np.ndarray,
               dfdt_ie:np.ndarray,
               gmtimes:np.ndarray,
               dfdt_gm:np.ndarray) -> None:
    ax.fill_between(ietimes,cpcp,fc='grey',label='CPCP')
    ax.plot(ietimes,dfdt_ie,c='orange',label=r'$d\phi/dt$ IE')
    ax.plot(gmtimes,dfdt_gm,c='purple',label=r'$d\phi/dt$ GM')

def plot_cpcp_vs(lobes:pd.DataFrame,ie:dict) -> None:
    times = ie['time']
    cpcp_n = (np.max(ie['PHI [kV]'],axis=1)-
                np.min(ie['PHI [kV]'],axis=1))

    flux = get_normal_flux(ie,ie)
    flux_n = flux[0,:]
    dfdt_n1 = abs(central_difference(flux_n))/1000
    dfdt_n2 = np.convolve(dfdt_n1,np.ones(30)/30,mode='same')

    flux_lobe = lobes['Bf_injectionK3 [Wb]'].values/6.371**2
    dfdt_lobe = abs(central_difference(flux_lobe))/1000
    dfdt_lobe2 = np.convolve(dfdt_lobe,np.ones(30)/30,mode='same')

    '''
    EK1 = np.convolve(mp['K_netK1 [W]']+mp['UtotM1 [W]'],
                      np.ones(10)/10,mode='same')/-1e12
    EK5 = np.convolve(mp['K_netK5 [W]']+mp['UtotM5 [W]'],
                      np.ones(10)/10,mode='same')/-1e12
    EK = np.convolve(mp['K_net [W]']+mp['UtotM [W]'],
                      np.ones(10)/10,mode='same')/-1e12
    '''

    fig,ax = plt.subplots(1,1,figsize=[18,9])
    draw_cpcp_panel(ax,times,cpcp_n,dfdt_n2,lobes.index,dfdt_lobe2)
    general_plot_settings(ax,legend=True,timedelta=False,
                          xlim=[times[0],times[-1]])
    ax.set_xlabel('Time May 2024 [dy-hr]')
    ax.set_ylabel('Potential [kV]')

    # Save
    figurename = f"../outputs/figures/unfiled/cpcp_vs_dfdt.png"
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)

def plot_Ein_JH(mp:pd.DataFrame,ien:dict,ies:dict) -> None:
    times = ien['time']
    joule_heating_n = np.sum(ien['JouleHeat [mW/m^2]']*ies['Area [Re^2]']
                             *(6371*1e3)**2/1000/1e12,axis=1)
    joule_heating_s = np.sum(ies['JouleHeat [mW/m^2]']*ies['Area [Re^2]']
                             *(6371*1e3)**2/1000/1e12,axis=1)
    joule_heating = joule_heating_n + joule_heating_s
    Einput = np.convolve(mp['K_netK1 [W]']+mp['UtotM1 [W]'],
                                            np.ones(10)/10,mode='same')/-1e12
    fig,ax = plt.subplots(1,1,figsize=[18,9])
    ax.plot(times,joule_heating,c='red',label='Joule Heating')
    ax.plot(mp.index,Einput,c='blue',label='Energy Injection')
    general_plot_settings(ax,legend=True,timedelta=False,
                          xlim=[times[0],times[-1]])
    ax.set_xlabel('Time May 2024 [dy-hr]')
    ax.set_ylabel('Integrated Power [TW]')

    # Save
    figurename = f"../outputs/figures/unfiled/Ein_vs_JH.png"
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)

def main() -> None:
    # BATRSUS analysis
    print('Loading BATS analysis results ...')
    bats_analysis_data = load_hdf_sort(inAnalysis+'energetics.h5')
    mp = bats_analysis_data['mpdict']['ms_full']
    closed = bats_analysis_data['msdict']['closed']
    lobes = bats_analysis_data['msdict']['lobes']
    plasmasheet = bats_analysis_data['msdict']['plasmasheet']
    inner = bats_analysis_data['inner_mp']

    # IE solution
    print('Loading IE full solution:')
    print('\tNorth ...')
    ie_solution_N =dict(np.load(f"../data/large/IE/ionosphere/compiled_N.npz",
                                 allow_pickle=True))
    print('\tSouth ...')
    ie_solution_S =dict(np.load(f"../data/large/IE/ionosphere/compiled_S.npz",
                                 allow_pickle=True))

    #plot_cpcp_vs(lobes,ie_solution_N)
    plot_Ein_JH(mp,ie_solution_N,ie_solution_S)

if __name__ == "__main__":
    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='print'))

    global inBase,inAnalysis

    inBase = os.path.realpath('..')+'/'
    inAnalysis = os.path.join(inBase,'data/analysis/')

    main()
