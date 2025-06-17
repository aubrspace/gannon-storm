#!/usr/bin/env python3
"""Analyze and plot data from the Gannon storm
"""
import os,sys,glob,time
import numpy as np
from numpy import sin,cos,deg2rad,rad2deg,pi
import scipy
import statsmodels.api as sm
import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt
from cmcrameri import cm
#
from global_energetics.analysis.plot_tools import (pyplotsetup,
                                                   bin_and_describe,
                                                   extended_fill_between,
                                                   general_plot_settings)
from global_energetics.analysis.proc_satellites import(determine_satelliteIDs,
                                                       simdata_to_df,
                                                       add_derived_variables,
                                                       split_themis)

from global_energetics.analysis.proc_indices import read_indices,read_pc
from global_energetics.analysis.proc_hdf import load_hdf_sort
from global_energetics.extract.shue import r0_alpha_1998
from global_energetics.analysis.proc_ampere import read_currents

def plot_indices(sw,log,omni,pc,path):
    #setup figure
    indices,(axis1,axis2,axis3,axis4) =plt.subplots(4,1,figsize=[20,20],
                                                    sharex=True)
    axis1.fill_between(sw.index,np.sqrt(sw['bx']**2+sw['by']**2+sw['bz']**2),
                       label='|B|',fc='grey')
    axis1.plot(sw.index,sw['bx'],label='Bx',c='red')
    axis1.plot(sw.index,sw['by'],label='By',c='blue')
    axis1.plot(sw.index,sw['bz'],label='Bz',c='cyan',lw=3)
    axis2.plot(omni.index,omni['sym_h'],label='Sym-H',c='black',lw=3)
    axis2.plot(log.index,log['dst_sm'],label='SWMF',c='magenta',lw=1.5)
    axis3.fill_between(sw.index,sw['EinWang']/1e12,label='Ein_Wang2014',
                       fc='purple')
    axis4.plot(pc.index,pc['cpcpn'],label='Ridley&Kihn',c='black',lw=3)
    axis4.plot(pc.index,pc['cpcps'],label='_Ridley&Kihn',c='black',lw=1.5,
                                                                      ls='--')
    axis4.plot(log.index,log['cpcpn'],label='SWMF_N',c='magenta',lw=3)
    axis4.plot(log.index,log['cpcps'],label='_SWMF_S',c='magenta',lw=1.5,
                                                                      ls='--')

    for ax in [axis1,axis2,axis3,axis4]:
        ax.margins(x=0.01)
        #ax.axvline(TIMPACT,c='blue')
        #ax.axvline(TDIVERGE,c='red')
    general_plot_settings(axis1,do_xlabel=False,legend=True,
                          xlim=[TINIT,TEND],
                          ylabel=r'B $\left[nT\right]$',timedelta=False)
    axis1.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                     ncol=4, fancybox=True, shadow=True)
    general_plot_settings(axis2,do_xlabel=False,legend=True,
                    ylabel=r'$\Delta B$ $\left[nT\right]$',timedelta=False)
    general_plot_settings(axis3,do_xlabel=False,legend=True,
                         ylabel=r'Power $\left[TW\right]$',timedelta=False)
    general_plot_settings(axis4,do_xlabel=True,legend=True,
                          ylim=[0,700],
                          ylabel=r'CPCP $\left[kV\right]$',timedelta=False)
    indices.tight_layout(pad=1)
    figurename = f'{path}/indices.png'
    indices.savefig(figurename)
    plt.close(indices)
    print('\033[92m Created\033[00m',figurename)

def plot_energy_flux(mp,sw,omni,log,path):
    # Set window
    window = [TINIT,TEND]
    # Make plots for comparisons
    eflux,(ax_Eext,ax_Eint,ax_dst,ax_cpcp) =plt.subplots(4,1,figsize=[30,30],
                                                       sharex=True)
    # Energy flux at the external interfaces (SW-M coupling)
    ax_Eext.fill_between(mp.index,mp['K_netK1 [W]']/1e12,label='K1 (OpenMP)',
                         fc='blue')
    ax_Eext.fill_between(mp.index,mp['K_netK5 [W]']/1e12,label='K5 (ClosedMP)',
                         fc='red')
    ax_Eext.fill_between(mp.index,(mp['K_netK1 [W]']+mp['K_netK5 [W]'])/1e12,
                         label='K1+K5 (MP)',fc='lightgrey',alpha=0.4)
    ax_Eext.plot(sw.index,-sw['EinWang']/1e12,
                   c='black',lw=4,label='Wang2014')
    # Energy flux at the internal interfaces (M-I coupling)
    ax_Eint.fill_between(closed.index,closed['K_netK2a [W]']/1e12,
                         label='K2a (Cusp)',fc='magenta',alpha=0.8)
    ax_Eint.fill_between(closed.index,closed['K_netK2b [W]']/1e12,
                         label='K2b (Plasmasheet)',fc='dodgerblue',alpha=0.8)
    ax_Eint.plot(lobes.index,lobes['K_netK3 [W]']/1e12,
                         label='K3 (OpenIB)',c='darkslategrey',lw=3)
    ax_Eint.plot(closed.index,closed['K_netK7 [W]']/1e12,lw=3,
                         label='K7 (ClosedIB)',c='orange')
    # Sym-h and total energy content
    U0 = mp['Utot [J]'].loc[mp.index[0]]
    ax_dst.fill_between(mp.index,(mp['Utot [J]']-U0)/(1.5*-8e13),label='Utot',
                        fc='grey')
    ax_dst.plot(omni.index,omni['sym_h'],label='Sym-H',c='black',lw=3)
    ax_dst.plot(log.index,log['dst_sm'],label='Dst',c='purple',lw=1.5)
    # Cross Polar Cap Potential (internal convection)
    ax_cpcp.plot(log.index,log['cpcpn'],label='North',c='blue',lw=3)
    ax_cpcp.plot(log.index,log['cpcps'],label='South',c='red',lw=1.5)

    # Decorations
    general_plot_settings(ax_Eext,do_xlabel=False,legend=True,
                          xlim=window,
                          ylim=[-100,30],
                          ylabel=r'Power $\left[TW\right]$',
                          timedelta=False)
    ax_Eext.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                     ncol=5, fancybox=True, shadow=True)
    general_plot_settings(ax_Eint,do_xlabel=False,legend=True,
                          xlim=window,
                          legend_loc='lower right',
                          ylim=[-75,30],
                          ylabel=r'Power $\left[TW\right]$',
                          timedelta=False)
    general_plot_settings(ax_dst,do_xlabel=False,legend=True,
                          xlim=window,
                          ylabel=r'Energy $\left[PJ\right]$',
                          timedelta=False)
    general_plot_settings(ax_cpcp,do_xlabel=True,legend=True,
                          xlim=window,
                          ylabel=r'CPCP $\left[kV\right]$',
                          timedelta=False)
    # save
    eflux.tight_layout(pad=1)
    figurename = f'{path}/eflux.png'
    eflux.savefig(figurename)
    plt.close(eflux)
    print('\033[92m Created\033[00m',figurename)

def plot_pc_size(dataset:dict,path:str) -> None:
    #
    I_swmf  = dataset['analysis']['currents']
    I_ampere = dataset['ampere']
    pc = dataset['obs2']['pc']

    # Make plots for comparisons
    pc_fig,(ax_lat,ax_area) =plt.subplots(2,1,figsize=[20,15],sharex=True)

    # Lat points for noon/midnight in each hemisphere
    ax_lat.plot(I_swmf.index,90-I_swmf['theta_day_N'],label='CMEE Noon',
                c='gold')
    ax_lat.plot(I_swmf.index,90-I_swmf['theta_night_N'],label='CMEE Midnight',
                c='purple')
    ax_lat.plot(I_swmf.index,I_swmf['theta_day_S']-90,label='_CMEE Noon',
                c='gold',ls='--')
    ax_lat.plot(I_swmf.index,I_swmf['theta_night_S']-90,
                label='_CMEE Midnight',c='purple',ls='--')
    # Lat points for noon/midnight of ocflb
    ax_lat.plot(I_swmf.index,90-I_swmf['theta_noonN'],label='OCFLB Noon',
                c='red')
    ax_lat.plot(I_swmf.index,90-I_swmf['theta_midnightN'],
                label='OCFLB Midnight',c='blue')
    #ax_lat.plot(I_swmf.index,90-I_swmf['theta_minN'],label='_OCFLBmin',
    #            c='lightgrey')
    ax_lat.plot(I_swmf.index,I_swmf['theta_noonS']-90,label='_OCFLB Noon',
                c='red',ls='--')
    ax_lat.plot(I_swmf.index,I_swmf['theta_midnightS']-90,
                label='_OCFLB Midnight',c='blue',ls='--')
    #ax_lat2.plot(I_swmf.index,I_swmf['theta_minS']-90,label='_OCFLBmin',
    #            c='lightgrey')

    # Area of the polar cap according to different sources
    ax_area.plot(I_swmf.index,I_swmf['Aoval_N'],label='CMEE Area',c='gold')
    ax_area.plot(I_swmf.index,I_swmf['open_areaN'],label='OCFLB Area',c='red')
    ax_area.plot(I_swmf.index,I_swmf['Aoval_S'],label='_CMEE Area',
                  c='gold',ls='--')
    ax_area.plot(I_swmf.index,I_swmf['open_areaS'],label='_OCFLB Area',
                  c='red',ls='--')

    # decorations
    general_plot_settings(ax_lat,do_xlabel=False,legend=True,
                          xlim=[TINIT,TEND],
                          ylabel=r'Latitude $\left[^{\circ}\right]$',
                          timedelta=False)
    general_plot_settings(ax_area,do_xlabel=True,legend=True,
                          xlim=[TINIT,TEND],
                          ylabel=r'Area $\left[{R_e}^2\right]$',
                          timedelta=False)

    ax_lat.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                    ncol=4, fancybox=True, shadow=True)

    # save
    pc_fig.tight_layout(pad=1)
    figurename = f'{path}/polar_cap.png'
    pc_fig.savefig(figurename)
    plt.close(pc_fig)
    print('\033[92m Created\033[00m',figurename)

def plot_standoff(mp,sw,sw2,path):
    # Make plots for comparisons
    ni_v_ne,(ax_n,ax_pdyn,ax_rmp,ax_Ein) =plt.subplots(4,1,figsize=[20,20],
                                                       sharex=True)
    # density comparing ne vs ni
    ax_n.plot(sw.index,sw['density'],label='ni',c='black')
    ax_n.plot(sw2.index,sw2['density'],label='ne',c='goldenrod')
    # effect on dynamic pressure
    ax_pdyn.plot(sw.index,sw['pdyn'],label='ni',c='black')
    ax_pdyn.plot(sw2.index,sw2['pdyn'],label='ne',c='goldenrod')
    # effect on shue standoff distance + compare w/ sim
    ax_rmp.plot(sw.index,sw['r_shue98'],label='ni',c='black')
    ax_rmp.plot(sw2.index,sw2['r_shue98'],label='ne',c='goldenrod')
    ax_rmp.plot(mp.index,mp['X_subsolar [Re]'],label='sim',c='purple')
    ax_rmp.axhline(3,c='red',lw=3,ls='--')
    ax_rmp.axvline(TDIVERGE,c='red')
    # effect on Einput + compare w/ sim
    ax_Ein.plot(sw.index,-sw['EinWang']/1e12,label='ni',c='black')
    ax_Ein.plot(sw2.index,-sw2['EinWang']/1e12,label='ne',c='goldenrod')
    ax_Ein.plot(mp.index,mp['K_netK1 [W]']/1e12,label='sim',c='purple')
    ax_Ein.axvline(TDIVERGE,c='red')

    # decorations
    general_plot_settings(ax_n,do_xlabel=False,legend=True,
                          xlim=[TIMPACT-dt.timedelta(minutes=30),
                                TIMPACT+dt.timedelta(hours=24)],
                          ylabel=r'Density $\left[\#/cc\right]$',
                          timedelta=False)
    general_plot_settings(ax_pdyn,do_xlabel=False,legend=True,
                          ylabel=r'$P_{dyn}\left[nPa\right]$',
                          timedelta=False)
    general_plot_settings(ax_rmp,do_xlabel=False,legend=True,
                          ylabel=r'$X_{standoff}\left[R_E\right]$',
                          timedelta=False)
    general_plot_settings(ax_Ein,do_xlabel=True,legend=True,
                          ylabel=r'$E_{input}\left[TW\right]$',
                          timedelta=False)
    # Make plots for comparisons

    # save
    ni_v_ne.tight_layout(pad=1)
    figurename = f'{path}/ni_vs_ne.png'
    ni_v_ne.savefig(figurename)
    plt.close(ni_v_ne)
    print('\033[92m Created\033[00m',figurename)

def plot_satellites(themisA,themisD,themisE,
                    old_themisA,old_themisD,old_themisE,
                    outPath):
    # Make plots for comparisons
    fig1,(ax_vx) =plt.subplots(1,1,figsize=[24,15],
                                                       sharex=True)
    # X velocity
    ax_vx.scatter(themisA.index,themisA['ux'],label='ThemisA',s=30,c='black')
    ax_vx.scatter(themisD.index,themisD['ux'],label='ThemisD',s=30,c='blue')
    ax_vx.scatter(themisE.index,themisE['ux'],label='ThemisE',s=30,c='gold')

    '''
    ax_vx.plot(old_themisA.index,old_themisA['U_x [km/s]'],label='TA-firstRun',
                  c='black',marker='x')
    ax_vx.plot(old_themisD.index,old_themisD['U_x [km/s]'],label='TD-firstRun',
                  c='blue',marker='x')
    ax_vx.plot(old_themisE.index,old_themisE['U_x [km/s]'],label='TE-firstRun',
                  c='gold',marker='x')
    '''

    # decorations
    general_plot_settings(ax_vx,do_xlabel=True,legend=True,
                          xlim=[TIMPACT-dt.timedelta(minutes=30),
                                TIMPACT+dt.timedelta(minutes=120)],
                          ylabel=r'$V_{X}\left[km/s\right]$',
                          timedelta=False)

    # save
    fig1.tight_layout(pad=1)
    figurename = f'{path}/sat_ux.png'
    fig1.savefig(figurename)
    plt.close(fig1)
    print('\033[92m Created\033[00m',figurename)

def plot_data_compare(dataset: dict,outPath: dict) -> None:
    I_swmf  = dataset['analysis']['currents']
    I_ampere = dataset['ampere']
    # Create Figures and Plots
    fig1, ax1 = plt.subplots(figsize=[22,15])
    fig2, [ax2a,ax2b] = plt.subplots(2,1,figsize=[22,18])

    ## Draw plots
    # Ax1
    ax1.plot(I_swmf.index,I_swmf['up_north_MA'],label='SWMF_N',
             c='blue')
    ax1.plot(I_swmf.index,-I_swmf['down_north_MA'],label='_SWMF_N_down',
             c='blue',ls='--')
    ax1.plot(I_swmf.index,I_swmf['up_south_MA'],label='SWMF_S',
             c='purple')
    ax1.plot(I_swmf.index,-I_swmf['down_south_MA'],label='_SWMF_S_down',
             c='purple',ls='--')

    ax1.plot(I_ampere.index,I_ampere['I_total_up_North_[MA]'],
             label='AMPERE_N',c='red')
    ax1.plot(I_ampere.index,I_ampere['I_total_down_North_[MA]'],
             label='_AMPERE_N_down',c='red',ls='--')
    ax1.plot(I_ampere.index,I_ampere['I_total_up_South_[MA]'],
             label='AMPERE_S',c='orange')
    ax1.plot(I_ampere.index,I_ampere['I_total_down_South_[MA]'],
             label='_AMPERE_S_down',c='orange',ls='--')

    # Ax2a
    ax2a.plot(I_swmf.index,I_swmf['UP_R1_N'],label='SWMF_N',
             c='blue')
    ax2a.plot(I_swmf.index,I_swmf['DOWN_R1_N'],label='_SWMF_N_down',
             c='blue',ls='--')
    ax2a.plot(I_swmf.index,I_swmf['UP_R1_S'],label='SWMF_S',
             c='purple')
    ax2a.plot(I_swmf.index,I_swmf['DOWN_R1_S'],label='_SWMF_S_down',
             c='purple',ls='--')
    # Ax2b
    ax2b.plot(I_swmf.index,I_swmf['UP_R2_N'],label='SWMF_N',
             c='blue')
    ax2b.plot(I_swmf.index,I_swmf['DOWN_R2_N'],label='_SWMF_N_down',
             c='blue',ls='--')
    ax2b.plot(I_swmf.index,I_swmf['UP_R2_S'],label='SWMF_S',
             c='purple')
    ax2b.plot(I_swmf.index,I_swmf['DOWN_R2_S'],label='_SWMF_S_down',
             c='purple',ls='--')


    # Decorate Plots
    general_plot_settings(ax1,do_xlabel=True,legend=False,
                          xlabel=r'Time $\left[Day-Hr\right]$',
                          ylabel=r'$\int$FAC $\left[MA\right]$',
                          xlim=[TINIT,TEND], timdelta=False)
    ax1.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                    ncol=4, fancybox=True, shadow=True)

    general_plot_settings(ax2a,do_xlabel=True,legend=False,
                          xlabel=r'Time $\left[Day-Hr\right]$',
                          ylabel=r'$\int$R1 $\left[MA\right]$',
                          xlim=[TINIT,TEND], timdelta=False)
    ax2a.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                    ncol=4, fancybox=True, shadow=True)

    general_plot_settings(ax2b,do_xlabel=True,legend=False,
                          xlabel=r'Time $\left[Day-Hr\right]$',
                          ylabel=r'$\int$R2 $\left[MA\right]$',
                          xlim=[TINIT,TEND], timdelta=False)
    ax2b.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05),
                    ncol=4, fancybox=True, shadow=True)

    # Save Plots
    fig1.tight_layout(pad=1)
    figurename = path+'/ampere_compare_1.png'
    fig1.savefig(figurename)
    plt.close(fig1)
    print('\033[92m Created\033[00m',figurename)

    fig2.tight_layout(pad=1)
    figurename = path+'/ampere_compare_2.png'
    fig2.savefig(figurename)
    plt.close(fig2)
    print('\033[92m Created\033[00m',figurename)


def plot_saturation(mp_test,dataset,outPath):
    # Basic data wrangling for the test set of data
    # Get data to a common time axis
    index_log = dataset['obs2']['swmf_log'].index
    index_sw = dataset['obs2']['swmf_sw'].index
    t_log = [float(t.to_numpy()) for t in index_log-TMIN]
    t_sw = [float(t.to_numpy()) for t in index_sw-TMIN]
    t_test = [float(t.to_numpy()) for t in mp_test.index-TMIN]
    # Extract the quantities for this subset of data
    inner_test = dataset['analysis']['inner_mp']
    closed_test = dataset['analysis']['msdict']['closed']
    lobes_test = dataset['analysis']['msdict']['lobes']
    #K1   = (mp_test['K_netK1 [W]']+mp_test['UtotM1 [W]']).resample('300s').mean()/-1e12
    #K1   = (mp_test['K_netK1 [W]']+mp_test['UtotM1 [W]'])/-1e12
    K1   = (mp_test['K_netK1 [W]']+mp_test['UtotM1 [W]']
                                                ).rolling('600s').mean()/-1e12
    K    = (mp_test['K_netK1 [W]']+mp_test['UtotM1 [W]']+
            mp_test['K_netK5 [W]']+mp_test['UtotM5 [W]']+
           closed_test['K_netK7 [W]']+lobes_test['K_netK3 [W]'])/-1e12
    U    = (mp_test['Utot [J]'])/1e15
    Ein  = pd.Series(index=K1.index,
                     data=np.interp(t_test,t_sw,
                           dataset['obs2']['swmf_sw']['EinWang'].values/1e12))
    Esw  = pd.Series(index=K1.index,
                     data=np.interp(t_test,t_sw,
                                dataset['obs2']['swmf_sw']['Esw'].values/1e3))
    BOYLE = pd.Series(index=K1.index,
                     data=np.interp(t_test,t_sw,
                           dataset['obs2']['swmf_sw']['CPCP_B97'].values))
    SHILL = pd.Series(index=K1.index,
                     data=np.interp(t_test,t_sw,
                           dataset['obs2']['swmf_sw']['CPCP_S02'].values))
    KRID = pd.Series(index=K1.index,
                     data=np.interp(t_test,t_sw,
                           dataset['obs2']['swmf_sw']['CPCP_K08'].values))
    CPCP = pd.Series(index=K1.index,
                     data=np.interp(t_test,t_log,
                                 dataset['obs2']['swmf_log']['cpcpn'].values))
    FAC  = (dataset['analysis']['currents']['up_north_MA']+
            #dataset['analysis']['currents']['up_south_MA']+
            dataset['analysis']['currents']['down_north_MA'])
            #dataset['analysis']['currents']['down_south_MA'])
    MA   = pd.Series(index=K1.index,
                     data=np.interp(t_test,t_sw,
                                 dataset['obs2']['swmf_sw']['Ma'].values))
    #.resample('300s').mean()
    t_ie   = [float(t.to_numpy()) for t in FAC.index-TMIN]
    FAC  = pd.Series(index=K1.index,data=np.interp(t_test,t_ie,FAC.values))
    R1   = (abs(dataset['analysis']['currents']['UP_R1_N'])+
            abs(dataset['analysis']['currents']['DOWN_R1_N']))
            #abs(dataset['analysis']['currents']['UP_R1_S'])+
            #abs(dataset['analysis']['currents']['DOWN_R1_N']))
    #.resample('300s').mean()
    R1  = pd.Series(index=K1.index,data=np.interp(t_test,t_ie,R1.values))
    R2   = (abs(dataset['analysis']['currents']['UP_R2_N'])+
            abs(dataset['analysis']['currents']['DOWN_R2_N']))
            #abs(dataset['analysis']['currents']['UP_R2_S'])+
            #abs(dataset['analysis']['currents']['DOWN_R2_N']))
    #.resample('300s').mean()
    R2  = pd.Series(index=K1.index,data=np.interp(t_test,t_ie,R2.values))
    Upoints = np.linspace(20,85,100)
    decay_points = Upoints*1e3/(60*60*10)
    voltages = dataset['analysis']['voltages']
    V = pd.DataFrame()
    Vmatch = pd.DataFrame()
    for key in voltages.keys():
        if all(voltages[key].isna()):
            continue
        t_v = [float(t.to_numpy()) for t in
                     dataset['analysis']['voltages'][key].dropna().index-TMIN]
        series = pd.Series(index=K1.index,data=np.interp(t_test,t_v,
                           voltages[key].dropna().values))
        s5min  = abs(series).resample('300s').mean()
        V[key] = s5min
        t_match = [float(t.to_numpy()) for t in V[key].index-TMIN]
        Vmatch[key] = pd.Series(index=K1.index,data=np.interp(t_test,t_match,
                                                               V[key].values))
    #TODO
    #   Make nice plot of Vmatch (mean of null values?) vs:
    #       FAC
    #       CPCP
    #       Esw
    #       Esw*mp_width
    #   Calculate distance across max/min potentials
    #       color above maps by this distance
    #       compare that distance vs mp_width
    #   Figure out how to separate out R1 and R2 currents
    #       in the IE results:
    #           try finding countours around zero potential
    #           take the two 'inner most' as R1
    #           take the two 'outer most' as R2
    #           split any remaining 'unknown' across both, but record
    #   Finally:
    #       See if Esw   <-> FAC_R1 is 1-1
    #              V_eff <-> FAC_R1 is 1-1
    #              R2    <-> Utot   is 1-1 (or maybe some other observable?)
    #              Esw   <-> CPCP is 1-1 given an FAC_R2 level
    # Initialize things for context data
    T0 = dt.datetime(2022,6,6,0,0)
    allK1 = np.array([])
    allK = np.array([])
    allEin,allEsw,allMa = np.array([]),np.array([]),np.array([])
    allCPCP = np.array([])
    allU    = np.array([])
    allFAC    = np.array([])
    testpoints = ['stretched_LOWnLOWu',
                  'stretched_MEDnLOWu',
                  'stretched_HIGHnLOWu',
                  'stretched_LOWnMEDu',
                  'stretched_MEDnMEDu',
                  'stretched_HIGHnMEDu',
                  'stretched_LOWnHIGHu',
                  'stretched_MEDnHIGHu',
                  'stretched_HIGHnHIGHu']
    for i,run in enumerate(testpoints):
        if run not in dataset.keys():
            continue
        # More data wrangling over the missing data times in the context data
        tstart = T0+dt.timedelta(minutes=10)
        mp = dataset[run]['mpdict']['ms_full'][
                               dataset[run]['mpdict']['ms_full'].index>tstart]
        mp = mp.resample('60s').asfreq()
        inner = dataset[run]['inner_mp'][
                               dataset[run]['inner_mp'].index>tstart]
        inner = inner.resample('60s').asfreq()
        closed = dataset[run]['msdict']['closed'][
                               dataset[run]['msdict']['closed'].index>tstart]
        closed = closed.resample('60s').asfreq()
        lobes = dataset[run]['msdict']['lobes'][
                               dataset[run]['msdict']['lobes'].index>tstart]
        lobes = lobes.resample('60s').asfreq()
        # Get data to a common time axis
        index_log = dataset[run]['obs']['swmf_log'].index
        index_sw = dataset[run]['obs']['swmf_sw'].index
        index_fac = dataset[run]['currents'].index
        t_log = [float(t.to_numpy()) for t in index_log-T0]
        t_sw = [float(t.to_numpy()) for t in index_sw-T0]
        t_energy = [float(t.to_numpy()) for t in mp.index-T0]
        t_fac = [float(t.to_numpy()) for t in index_fac-T0]

        # Extract the quantities for this subset of data
        evEin   = np.interp(t_energy,t_sw,
                        dataset[run]['obs']['swmf_sw']['EinWang'].values/1e12)
        evEsw   = np.interp(t_energy,t_sw,
                        dataset[run]['obs']['swmf_sw']['Esw'].values/1e3)
        evMa    = np.interp(t_energy,t_sw,
                        dataset[run]['obs']['swmf_sw']['Ma'].values)
        evCPCP  = np.interp(t_energy,t_log,
                        dataset[run]['obs']['swmf_log']['cpcpn'].values)
        #evK1    = (mp['K_netK1 [W]']+mp['UtotM1 [W]'])/-1e12
        evK1    = (mp['K_netK1 [W]']+mp['UtotM1 [W]']
                                                ).rolling('600s').mean()/-1e12
        evK     = (mp['K_netK1 [W]']+mp['UtotM1 [W]']+
                   mp['K_netK5 [W]']+mp['UtotM5 [W]']-
                   closed['K_netK7 [W]']+lobes['K_netK3 [W]'])/-1e12
        evU     = (mp['Utot [J]'])/1e15
        evFAC = np.interp(t_energy,t_fac,
                                    dataset[run]['currents']['up_north_MA']+
                                    dataset[run]['currents']['down_north_MA'])
        # Append subset of data to a full data array for further vis
        allK1       = np.append(allK1,evK1.values)
        allK        = np.append(allK,evK.values)
        allEin      = np.append(allEin,evEin)
        allEsw      = np.append(allEsw,evEsw)
        allMa       = np.append(allMa,evMa)
        allCPCP     = np.append(allCPCP,evCPCP)
        allU        = np.append(allU,evU)
        allFAC      = np.append(allFAC,evFAC)
    df_reference = pd.DataFrame({'Ein':allEin,
                               'Esw':allEsw,
                               'Ma':allMa,
                               'CPCP':allCPCP,
                               'U':allU,
                               'K1':allK1,
                               'K':allK,
                               'FAC':allFAC})
    df_reference = df_reference.dropna()
    # Obtain low,50, and high %tiles, and variance binned by our X axis
    Ein_bins  = np.linspace(1,24,11)
    Esw_bins  = np.linspace(df_reference['Esw'].quantile(0.005),
                            df_reference['Esw'].quantile(0.995),11)
    CPCP_bins = np.linspace(df_reference['CPCP'].quantile(0.005),
                            df_reference['CPCP'].quantile(0.995),11)
    CPCPdict  = bin_and_describe(df_reference['CPCP'],df_reference['K1'],
                                 df_reference,CPCP_bins,0.05,0.95)
    Satdict   = bin_and_describe(df_reference['Ein'],df_reference['CPCP'],
                                 df_reference,Ein_bins,0.05,0.95)
    Satdict2  = bin_and_describe(df_reference['Esw'],df_reference['CPCP'],
                                 df_reference,Esw_bins,0.05,0.95)
    U_bins    = np.linspace(df_reference['U'].quantile(0.01),
                            df_reference['U'].quantile(0.99),11)
    Udict     = bin_and_describe(df_reference['U'],df_reference['K1'],
                                 df_reference,U_bins,0.05,0.95)
    K1_bins   = np.linspace(df_reference['K1'].quantile(0.01),
                            df_reference['K1'].quantile(0.99),11)
    K1dict    = bin_and_describe(df_reference['K1'],df_reference['U'],
                                 df_reference,K1_bins,0.05,0.95)
    K_bins    = np.linspace(df_reference['K'].quantile(0.01),
                            df_reference['K'].quantile(0.99),11)
    Kdict     = bin_and_describe(df_reference['K'],df_reference['U'],
                                 df_reference,K_bins,0.05,0.95)

    test_Ein_bins = np.linspace(Ein.quantile(0.005),Ein.quantile(0.995),33)
    test_Esw_bins = np.linspace(Esw.quantile(0.005),Esw.quantile(0.995),33)
    test_FAC_bins = np.linspace(FAC.quantile(0.005),FAC.quantile(0.995),33)
    test_U_bins   = np.linspace(U.quantile(0.005),U.quantile(0.995),33)
    test_K1_bins  = np.linspace(K1.quantile(0.005),K1.quantile(0.995),33)
    test_Satdict  = bin_and_describe(Ein,CPCP,CPCP,test_Ein_bins,0.05,0.95)
    test_Satdict2 = bin_and_describe(Esw,CPCP,CPCP,test_Esw_bins,0.05,0.95)
    test_Satdict3 = bin_and_describe(Ein,FAC,FAC,test_Ein_bins,0.05,0.95)
    test_Satdict4 = bin_and_describe(FAC,CPCP,CPCP,test_FAC_bins,0.05,0.95)
    test_Convert  = bin_and_describe(K1,FAC,FAC,test_K1_bins,0.05,0.95)
    test_Udict    = bin_and_describe(U,K1,K1,test_U_bins,0.05,0.95)

    # Create Figures and Plots
    fig1, ax1 = plt.subplots(figsize=[18,15])
    fig2, ax2 = plt.subplots(figsize=[18,15])
    fig3, ax3 = plt.subplots(figsize=[18,15])
    fig4, ax4 = plt.subplots(figsize=[18,15])
    fig5, ax5 = plt.subplots(figsize=[18,15])
    fig6, ax6 = plt.subplots(figsize=[18,15])
    fig7, [ax7_top,ax7_bot] = plt.subplots(2,1,figsize=[20,15],sharex=True)
    #fig8, [[axa,axb],[axc,axd],[axe,axf]] =plt.subplots(3,2)
    fig8, axa = plt.subplots()
    # Draw on Plots

    # Ax1
    extended_fill_between(ax1,Ein_bins,Satdict['pLow_all'],
                                       Satdict['pHigh_all'],'grey',0.2)
    extended_fill_between(ax1,test_Ein_bins,test_Satdict['pLow_all'],
                                            test_Satdict['pHigh_all'],
                                            'gold',0.2)
    ax1.plot(Ein_bins,Satdict['p50_all'],c='darkgrey',ls='--',lw=4)
    ax1.plot(test_Ein_bins,test_Satdict['p50_all'],c='black',lw=4)
    sc1 = ax1.scatter(Ein,CPCP,cmap=cm.managua,c=[t/3600e9 for t in t_test],
                      s=50,alpha=0.8)
    cbar1 = fig1.colorbar(sc1)
    ax1.scatter(df_reference['Ein'],df_reference['CPCP'],
                s=25,marker='x',c='grey',alpha=0.2)

    # Ax2
    extended_fill_between(ax2,Esw_bins,Satdict2['pLow_all'],
                                       Satdict2['pHigh_all'],'grey',0.2)
    extended_fill_between(ax2,test_Esw_bins,test_Satdict2['pLow_all'],
                                            test_Satdict2['pHigh_all'],
                                            'gold',0.2)
    ax2.plot(Esw_bins,Satdict2['p50_all'],c='darkgrey',ls='--',lw=4)
    ax2.plot(test_Esw_bins,test_Satdict2['p50_all'],c='black',lw=4)
    sc2 = ax2.scatter(Esw,CPCP,cmap=cm.managua,c=[t/3600e9 for t in t_test],
                      s=50,alpha=0.8)
    cbar2 = fig2.colorbar(sc2)
    ax2.scatter(df_reference['Esw'],df_reference['CPCP'],
                s=25,marker='x',c='grey',alpha=0.2)


    # Ax3
    extended_fill_between(ax3,U_bins,Udict['pLow_all'],
                                     Udict['pHigh_all'],'grey',0.2)
    extended_fill_between(ax3,test_U_bins,test_Udict['pLow_all'],
                                          test_Udict['pHigh_all'],
                                          'gold',0.2)
    ax3.plot(U_bins,Udict['p50_all'],c='darkgrey',ls='--',lw=4)
    ax3.plot(test_U_bins,test_Udict['p50_all'],c='black',lw=4)
    sc3 = ax3.scatter(U,K1,cmap=cm.managua,c=[t/3600e9 for t in t_test],
                      s=50,alpha=0.8)
    cbar3 = fig3.colorbar(sc3)
    ax3.scatter(df_reference['U'],df_reference['K1'],
                s=25,marker='x',c='grey',alpha=0.2)
    #ax3.plot(Upoints,decay_points,c='purple')

    # Ax4
    extended_fill_between(ax4,test_Ein_bins,test_Satdict3['pLow_all'],
                                            test_Satdict3['pHigh_all'],
                                            'gold',0.2)
    ax4.plot(test_Ein_bins,test_Satdict3['p50_all'],c='black',lw=4)
    #sc4 = ax4.scatter(Ein,FAC,cmap=cm.managua,c=[t/3600e9 for t in t_test],
    sc4 = ax4.scatter(Ein,FAC,cmap=cm.managua,c=MA,
                      s=50,alpha=0.8)
    cbar4 = fig4.colorbar(sc4)
    sc4b = ax4.scatter(df_reference['Ein'],df_reference['FAC'],
                #s=25,marker='x',c='grey',alpha=0.2)
                s=25,marker='x',c=df_reference['Ma'],alpha=0.2)
    cbar4b = fig4.colorbar(sc4b)

    # Ax5
    extended_fill_between(ax5,test_K1_bins,test_Convert['pLow_all'],
                                           test_Convert['pHigh_all'],
                                            'gold',0.2)
    ax5.plot(test_K1_bins,test_Convert['p50_all'],c='black',lw=4)
    sc5 = ax5.scatter(K1,FAC,cmap=cm.managua,c=[t/3600e9 for t in t_test],
                      s=50,alpha=0.8)
    cbar5 = fig5.colorbar(sc5)
    ax5.scatter(df_reference['K1'],df_reference['FAC'],
                s=25,marker='x',c='grey',alpha=0.2)
    model = sm.OLS(FAC.values,K1.values)
    results = model.fit()
    print(results.summary())
    names = ['Boyle1997','Sisco-Hill2002','Kivelson-Ridley2008']
    for i,predictor in enumerate([BOYLE,SHILL,KRID]):
        axa.scatter(Esw,predictor-CPCP,s=50,alpha=0.8,label=names[i])
    axa.set_xlabel(r'$E_{K-L}\left[mA/m\right]$')
    axa.set_ylabel(r'CPCP $\left[kV\right]$')
    axa.legend()
    #TODO
    #   Try out functional forms of Hill-Siscoe, Kivelson-Ridley, etc.
    #   Try out X^... [0.01, 0.05, 0.1, 0.15, 0.3, 0.5, 0.75, 1]
    #
    #   Tabulate (R^2, thumbnail of residuals) for K1->FAC & Ein->CPCP
    from IPython import embed; embed()
    slope,inter,r,p,std_err=scipy.stats.linregress(K1.values,FAC.values)
    ax5.plot(test_K1_bins,slope*test_K1_bins+inter,c='blue',ls='--',lw=3)
    ax5.text(0.02,0.84,r'$R^2$'+f'={r**2:.2f}',transform=ax5.transAxes,
                                   c='blue',horizontalalignment='left')

    slope,inter,r,p,std_err=scipy.stats.linregress(df_reference['K1'].values,
                                                   df_reference['FAC'].values)
    ax5.plot(K1_bins,slope*K1_bins+inter,c='red',ls='--',lw=3)
    ax5.text(0.02,0.94,r'$R^2$'+f'={r**2:.2f}',transform=ax5.transAxes,
                                   c='red',horizontalalignment='left')

    # Ax6
    extended_fill_between(ax6,test_FAC_bins,test_Satdict4['pLow_all'],
                                            test_Satdict4['pHigh_all'],
                                            'gold',0.2)
    ax6.plot(test_FAC_bins,test_Satdict4['p50_all'],c='black',lw=4)
    sc6 = ax6.scatter(FAC,CPCP,cmap=cm.managua,c=[t/3600e9 for t in t_test],
                      s=50,alpha=0.8)
    cbar6 = fig6.colorbar(sc6)
    ax6.scatter(df_reference['FAC'],df_reference['CPCP'],
                s=25,marker='x',c='grey',alpha=0.2)

    # Ax7
    ax7_top.plot(FAC.index,FAC,c='gold',lw=4,label='FAC')
    ax7_bot.plot(V.index,V['dV_Null_N'],c='black',label='Null_N')
    ax7_bot.plot(V.index,V['dV_Null_S'],c='grey',label='Null_N')
    ax7_bot.plot(V.index,V['dV_Bstream_N'],c='red',label='Jpar_N')
    ax7_bot.plot(V.index,V['dV_Bstream_S'],c='blue',label='Jpar_S')


    # Decorate Plots
    #ax1.set_xlim(0,25)
    #ax1.set_ylim(0,25)
    ax1.set_xlim(Ein.quantile(0.01),Ein.quantile(0.99))
    ax1.set_ylim(CPCP.quantile(0.01),CPCP.quantile(0.99))
    ax1.set_xlabel(r'$E_{in}\left[TW\right]$ Wang et al. 2014')
    ax1.set_ylabel(r'CPCP $\left[kV\right]$')
    cbar1.set_label(r'$\Delta t_{MIN}\left[min\right]$')

    ax2.set_xlim(Esw.quantile(0.01),Esw.quantile(0.99))
    ax2.set_ylim(CPCP.quantile(0.01),CPCP.quantile(0.99))
    ax2.set_xlabel(r'$E_{sw}\left[mV/m\right]$ Kan and Lee 1979')
    ax2.set_ylabel(r'CPCP $\left[kV\right]$')
    cbar2.set_label(r'$\Delta t_{MIN}\left[min\right]$')

    ax3.set_xlim(U.quantile(0.01),U.quantile(0.99))
    ax3.set_ylim(K1.quantile(0.01),K1.quantile(0.99))
    ax3.set_xlabel(r'$\int\mathbf{U}$ Energy $\left[PJ\right]$')
    ax3.set_ylabel(r'$\int\mathbf{K}_1$ Power $\left[TW\right]$')
    cbar3.set_label(r'$\Delta t_{MIN}\left[min\right]$')

    ax4.set_xlim(Ein.quantile(0.01),Ein.quantile(0.99))
    ax4.set_ylim(FAC.quantile(0.01),FAC.quantile(0.99))
    ax4.set_xlabel(r'$E_{in}\left[TW\right]$ Wang et al. 2014')
    ax4.set_ylabel(r'$\int$FAC $\left[MA\right]$')
    cbar4.set_label(r'$\Delta t_{MIN}\left[hr\right]$')

    ax5.set_xlim(K1.quantile(0.01),K1.quantile(0.99))
    ax5.set_ylim(FAC.quantile(0.01),FAC.quantile(0.99))
    ax5.set_xlabel(r'$\int\mathbf{K}_1$ Power $\left[TW\right]$')
    ax5.set_ylabel(r'$\int$FAC $\left[MA\right]$')
    cbar5.set_label(r'$\Delta t_{MIN}\left[hr\right]$')

    ax6.set_xlim(FAC.quantile(0.01),FAC.quantile(0.99))
    ax6.set_ylim(CPCP.quantile(0.01),CPCP.quantile(0.99))
    ax6.set_xlabel(r'$\int$FAC $\left[MA\right]$')
    ax6.set_ylabel(r'CPCP $\left[kV\right]$')
    cbar6.set_label(r'$\Delta t_{MIN}\left[hr\right]$')

    ax7_top.set_ylabel(r'Current $\left[MA\right]$')
    ax7_bot.set_xlim(dt.datetime(2024,5,10,14,0),
                     dt.datetime(2024,5,11,14,0))
    #ax7_bot.set_ylim([0,2000])
    ax7_bot.legend()
    ax7_bot.set_xlabel(r'Time')
    ax7_bot.set_ylabel(r'Voltage $\left[kV\right]$')



    # Save Plots
    fig1.tight_layout(pad=1)
    figurename = path+'/cpcp_saturation.png'
    fig1.savefig(figurename)
    plt.close(fig1)
    print('\033[92m Created\033[00m',figurename)

    fig2.tight_layout(pad=1)
    figurename = path+'/cpcp_saturation2.png'
    fig2.savefig(figurename)
    plt.close(fig2)
    print('\033[92m Created\033[00m',figurename)

    fig3.tight_layout(pad=1)
    figurename = path+'/energy_vs_K1.png'
    fig3.savefig(figurename)
    plt.close(fig3)
    print('\033[92m Created\033[00m',figurename)

    fig4.tight_layout(pad=1)
    figurename = path+'/cpcp_saturation3.png'
    fig4.savefig(figurename)
    plt.close(fig4)
    print('\033[92m Created\033[00m',figurename)

    fig5.tight_layout(pad=1)
    figurename = path+'/energy_conversion.png'
    fig5.savefig(figurename)
    plt.close(fig5)
    print('\033[92m Created\033[00m',figurename)

    fig6.tight_layout(pad=1)
    figurename = path+'/fac_conversion.png'
    fig6.savefig(figurename)
    plt.close(fig6)
    print('\033[92m Created\033[00m',figurename)

    fig7.tight_layout(pad=1)
    figurename = path+'/fac_vs_voltage.png'
    fig7.savefig(figurename)
    plt.close(fig7)
    print('\033[92m Created\033[00m',figurename)



if __name__ == "__main__":
    # Setup paths and key times
    #TINIT = dt.datetime(2024,5,9,6,0)
    TINIT = dt.datetime(2024,5,10,12,0)
    TIMPACT = dt.datetime(2024,5,10,17)
    TDIVERGE = dt.datetime(2024,5,10,19,30)
    TMIN  = dt.datetime(2024,5,11,1,30)
    TEND  = dt.datetime(2024,5,11,18)
    inBase = os.path.realpath('..')+'/'
    inLogs = os.path.join(inBase,'data/logs/')
    inSats = os.path.join(inBase,'data/sat/')
    inAnalysis = os.path.join(inBase,'data/analysis/')
    outPath = os.path.join(inBase,'outputs/figures')
    unfiled = os.path.join(outPath,'unfiled')
    for path in [outPath,unfiled,]:
        os.makedirs(path,exist_ok=True)
    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='print'))
    ## Log Data
    dataset = {}
    '''NOTE this is the old run w/ other settings
    dataset['obs'] = read_indices(inLogs+'temp/',start=TINIT,
                                  end=TIMPACT+dt.timedelta(hours=24),
                                  read_supermag=False)
    sw = dataset['obs']['swmf_sw']
    log = dataset['obs']['swmf_log']
    omni = dataset['obs']['omni']
    '''
    dataset['obs2'] = read_indices(inLogs,start=TINIT,
                                  end=TIMPACT+dt.timedelta(hours=24),
                                  read_supermag=False)
    dataset['obs2']['pc'] =read_pc(f"{inBase}data/pc_index/pcnpcs4318007.txt")

    sw2 = dataset['obs2']['swmf_sw']
    log2 = dataset['obs2']['swmf_log']
    omni = dataset['obs2']['omni']
    pc = dataset['obs2']['pc']
    ## Analysis Data
    dataset['analysis'] = load_hdf_sort(inAnalysis+'energetics.h5')
    with pd.HDFStore(inAnalysis+'integrated_currents.h5') as store:
        dataset['analysis']['currents'] = store['/FAC']

    mp = dataset['analysis']['mpdict']['ms_full']
    closed = dataset['analysis']['msdict']['closed']
    lobes = dataset['analysis']['msdict']['lobes']
    plasmasheet = dataset['analysis']['msdict']['plasmasheet']
    inner = dataset['analysis']['inner_mp']
    ## Reference Data
    events =[
             'stretched_LOWnLOWu',
             'stretched_LOWnMEDu',
             'stretched_LOWnHIGHu',
             #
             'stretched_HIGHnLOWu',
             'stretched_HIGHnMEDu',
             'stretched_HIGHnHIGHu',
             #
             'stretched_MEDnLOWu',
             'stretched_MEDnMEDu',
             'stretched_MEDnHIGHu',
             'stretched_LOWnLOWucontinued',
             ]
    for event in events:
        ideal_runs_path = '../../parameter_study/data'
        GMfile = os.path.join(f'{ideal_runs_path}/analysis/',event+'.h5')
        # GM data
        if os.path.exists(GMfile):
            dataset[event] = load_hdf_sort(GMfile)
        # Log data
        prefix = event.split('_')[1]+'_'
        dataset[event]['obs']=read_indices(f'{ideal_runs_path}/logs/',
                                            prefix=prefix,
                                        #start=dataset[event]['time'][0],
                 #end=dataset[event]['time'][-1]+dt.timedelta(seconds=1),
                                             read_supermag=False)
        # IE data
        FACfile = f"{event.split('stretched_')[1]}_integrated_currents.h5"
        with pd.HDFStore(f"{ideal_runs_path}/analysis/IE/{FACfile}") as store:
            dataset[event]['currents'] = store['/FAC']
    ## Satellite data
    with pd.HDFStore(f'{inBase}scripts/themis_plasma.h5') as store:
        thb_plasma = store['/themisB']
    ## Satellite data
    dataset['vsats1'] = {}
    with pd.HDFStore(f'{inSats}virtual_sats.h5') as store:
        for key in store.keys():
            dataset['vsats1'][key] = store[key]
    vsatfiles = glob.glob(f'{inSats}*.sat')
    dataset['vsats'] = simdata_to_df(vsatfiles)
    themisA = dataset['vsats']['themisA']
    themisD = dataset['vsats']['themisD']
    themisE = dataset['vsats']['themisE']
    old_themisA = dataset['vsats1']['/themisA']
    old_themisD = dataset['vsats1']['/themisD']
    old_themisE = dataset['vsats1']['/themisE']

    ## AMPERE data
    ampere_path = '../data/ampere'
    all_data = pd.DataFrame()
    for infile in glob.glob(f"{ampere_path}/*.dat"):
        df = read_currents(infile)
        all_data = pd.concat([all_data,df])
    all_data = all_data.replace(9999.00,np.nan)
    dataset['ampere'] = all_data

    ## Voltage data
    with pd.HDFStore(f'{inAnalysis}voltage_results.h5') as store:
        dataset['analysis']['voltages'] = store['/voltages']

    # Plot index results
    #plot_indices(sw,log,omni,outPath)
    #plot_indices(sw2,log2,omni,pc,outPath+'/unfiled/')

    # Plot standoff distance
    #plot_standoff(mp,sw,sw2,outPath)

    # Plot polar cap area
    #plot_pc_size(dataset,outPath+'/unfiled/')

    # Plot energy flux
    #plot_energy_flux(mp,sw2,omni,log2,outPath+'/unfiled/')

    # Plot sat data
    #plot_satellites(themisA,themisD,themisE,
    #                old_themisA,old_themisD,old_themisE,
    #                outPath)

    # Investigate energy input for saturation
    plot_saturation(mp,dataset,outPath)

    # Compare with data
    #plot_data_compare(dataset,outPath)
