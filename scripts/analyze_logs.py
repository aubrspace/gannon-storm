#!/usr/bin/env python3
"""Analyze and plot data for the parameter study of ideal runs
"""
import os,sys,glob,time
import numpy as np
from numpy import sin,cos,deg2rad,rad2deg,pi
import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt
#
from global_energetics.analysis.plot_tools import (pyplotsetup,
                                                   bin_and_describe,
                                                   extended_fill_between,
                                                   general_plot_settings)
from global_energetics.analysis.proc_satellites import(determine_satelliteIDs,
                                                       simdata_to_df,
                                                       add_derived_variables,
                                                       split_themis)

from global_energetics.analysis.proc_indices import read_indices
from global_energetics.analysis.proc_hdf import load_hdf_sort
from global_energetics.extract.shue import r0_alpha_1998

def plot_indices(sw,log,omni,path):
    #setup figure
    indices,(axis1,axis2,axis3,axis4) =plt.subplots(4,1,figsize=[20,20],
                                                    sharex=True)
    axis1.fill_between(sw.index,np.sqrt(sw['bx']**2+sw['by']**2+sw['bz']**2),
                       label='|B|',fc='grey')
    axis1.plot(sw.index,sw['bx'],label='Bx',c='red')
    axis1.plot(sw.index,sw['by'],label='By',c='blue')
    axis1.plot(sw.index,sw['bz'],label='Bz',c='cyan',lw=3)
    axis2.plot(log.index,log['cpcpn'],label='North',c='blue',lw=3)
    axis2.plot(log.index,log['cpcps'],label='South',c='red',lw=1.5)
    axis3.plot(omni.index,omni['sym_h'],label='Sym-H',c='black',lw=3)
    axis3.plot(log.index,log['dst_sm'],label='Dst',c='purple',lw=1.5)
    axis4.fill_between(sw.index,sw['EinWang']/1e12,label='Ein_Wang2014',
                       fc='purple')

    for ax in [axis1,axis2,axis3,axis4]:
        ax.margins(x=0.01)
        ax.axvline(TIMPACT,c='blue')
        ax.axvline(TDIVERGE,c='red')
    general_plot_settings(axis1,do_xlabel=False,legend=True,
                          xlim=[TIMPACT-dt.timedelta(hours=0.5),
                                TIMPACT+dt.timedelta(hours=24)],
                          ylabel=r'B $\left[nT\right]$',timedelta=False)
    general_plot_settings(axis2,do_xlabel=False,legend=True,
                          ylabel=r'CPCP $\left[kV\right]$',timedelta=False)
    general_plot_settings(axis3,do_xlabel=False,legend=True,
                    ylabel=r'$\Delta B$ $\left[nT\right]$',timedelta=False)
    general_plot_settings(axis4,do_xlabel=True,legend=True,
                         ylabel=r'Power $\left[TW\right]$',timedelta=False)
    indices.tight_layout(pad=1)
    figurename = f'{path}/indices.png'
    indices.savefig(figurename)
    plt.close(indices)
    print('\033[92m Created\033[00m',figurename)

def plot_energy_flux(mp,sw,omni,log,path):
    # Set window
    window = [TIMPACT-dt.timedelta(hours=2),
              TIMPACT+dt.timedelta(hours=24)]
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
    ax_dst.fill_between(mp.index,mp['Utot [J]']/(1.5*-8e13),label='Utot',
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
                     ncol=3, fancybox=True, shadow=True)
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

def plot_saturation(mp_test,dataset,outPath):
    # Basic data wrangling for the test set of data
    # Get data to a common time axis
    index_log = dataset['obs2']['swmf_log'].index
    index_sw = dataset['obs2']['swmf_sw'].index
    t_log = [float(t.to_numpy()) for t in index_log-TINIT]
    t_sw = [float(t.to_numpy()) for t in index_sw-TINIT]
    t_energy = [float(t.to_numpy()) for t in mp_test.index-TINIT]
    # Extract the quantities for this subset of data
    Ein  = np.interp(t_energy,t_sw,
                     dataset['obs2']['swmf_sw']['EinWang'].values/1e12)
    Ein2 = np.interp(t_energy,t_sw,
                    -dataset['obs2']['swmf_sw']['Pstorm'].values/1e12)
    CPCP = np.interp(t_energy,t_log,
                     dataset['obs2']['swmf_log']['cpcpn'].values)
    K1   = (mp_test['K_netK1 [W]']+mp_test['UtotM1 [W]'])/-1e12
    U    = (mp_test['Utot [J]'])/1e15
    # Initialize things for context data
    T0 = dt.datetime(2022,6,6,0,0)
    allK1 = np.array([])
    allEin,allEin2 = np.array([]),np.array([])
    allCPCP = np.array([])
    allU    = np.array([])
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
        mp = mp.resample('60S').asfreq()
        # Get data to a common time axis
        index_log = dataset[run]['obs']['swmf_log'].index
        index_sw = dataset[run]['obs']['swmf_sw'].index
        t_log = [float(t.to_numpy()) for t in index_log-T0]
        t_sw = [float(t.to_numpy()) for t in index_sw-T0]
        t_energy = [float(t.to_numpy()) for t in mp.index-T0]

        # Extract the quantities for this subset of data
        evEin   = np.interp(t_energy,t_sw,
                        dataset[run]['obs']['swmf_sw']['EinWang'].values/1e12)
        evEin2  = np.interp(t_energy,t_sw,
                        -dataset[run]['obs']['swmf_sw']['Pstorm'].values/1e12)
        evCPCP  = np.interp(t_energy,t_log,
                        dataset[run]['obs']['swmf_log']['cpcpn'].values)
        evK1    = (mp['K_netK1 [W]']+mp['UtotM1 [W]'])/-1e12
        evU     = (mp['Utot [J]'])/1e15
        # Append subset of data to a full data array for further vis
        allK1       = np.append(allK1,evK1.values)
        allEin      = np.append(allEin,evEin)
        allEin2     = np.append(allEin2,evEin2)
        allCPCP     = np.append(allCPCP,evCPCP)
        allU        = np.append(allU,evU)
    df_summary = pd.DataFrame({'Ein':allEin,
                               'Ein2':allEin2,
                               'CPCP':allCPCP,
                               'U':allU,
                               'K1':allK1})
    # Obtain low,50, and high %tiles, and variance binned by our X axis
    Ein_bins  = np.linspace(1,24,11)
    Eindict   = bin_and_describe(df_summary['Ein'],df_summary['K1'],
                               df_summary,Ein_bins,0.05,0.95)
    CPCP_bins = np.linspace(df_summary['CPCP'].quantile(0.005),
                            df_summary['CPCP'].quantile(0.995),11)
    CPCPdict  = bin_and_describe(df_summary['CPCP'],df_summary['K1'],
                                 df_summary,CPCP_bins,0.05,0.95)
    Satdict   = bin_and_describe(df_summary['Ein'],df_summary['CPCP'],
                                 df_summary,Ein_bins,0.05,0.95)
    U_bins    = np.linspace(df_summary['U'].quantile(0.01),
                            df_summary['U'].quantile(0.99),11)
    Udict     = bin_and_describe(df_summary['U'],df_summary['K1'],
                                 df_summary,U_bins,0.05,0.95)
    K_bins    = np.linspace(df_summary['K1'].quantile(0.01),
                            df_summary['K1'].quantile(0.99),11)
    Kdict     = bin_and_describe(df_summary['K1'],df_summary['U'],
                                 df_summary,K_bins,0.05,0.95)
    from IPython import embed; embed()
    pass

if __name__ == "__main__":
    # Setup paths and key times
    TINIT = dt.datetime(2024,5,9,6,0)
    TIMPACT = dt.datetime(2024,5,10,17)
    TDIVERGE = dt.datetime(2024,5,10,19,30)
    inBase = os.path.realpath('..')+'/'
    inLogs = os.path.join(inBase,'outputs/logs/')
    inSats = os.path.join(inBase,'outputs/sat/')
    inAnalysis = os.path.join(inBase,'outputs/analysis/')
    outPath = os.path.join(inBase,'outputs/figures')
    unfiled = os.path.join(outPath,'unfiled')
    for path in [outPath,unfiled,]:
        os.makedirs(path,exist_ok=True)
    #setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='print'))
    ## Log Data
    dataset = {}
    dataset['obs'] = read_indices(inLogs+'temp/',start=TINIT,
                                  end=TIMPACT+dt.timedelta(hours=24),
                                  read_supermag=False)
    sw = dataset['obs']['swmf_sw']
    log = dataset['obs']['swmf_log']
    omni = dataset['obs']['omni']
    dataset['obs2'] = read_indices(inLogs,start=TINIT,
                                  end=TIMPACT+dt.timedelta(hours=24),
                                  read_supermag=False)
    sw2 = dataset['obs2']['swmf_sw']
    log2 = dataset['obs2']['swmf_log']
    omni = dataset['obs2']['omni']
    ## Analysis Data
    dataset['analysis'] = load_hdf_sort(inAnalysis+'energetics.h5')
    mp = dataset['analysis']['mpdict']['ms_full']
    closed = dataset['analysis']['msdict']['closed']
    lobes = dataset['analysis']['msdict']['lobes']
    plasmasheet = dataset['analysis']['msdict']['plasmasheet']
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
        GMfile = os.path.join('../../parameter_study/data/analysis/',
                              event+'.h5')
        # GM data
        if os.path.exists(GMfile):
            dataset[event] = load_hdf_sort(GMfile)
        # Log data
        prefix = event.split('_')[1]+'_'
        dataset[event]['obs']=read_indices('../../parameter_study/data/logs/',
                                            prefix=prefix,
                                        #start=dataset[event]['time'][0],
                 #end=dataset[event]['time'][-1]+dt.timedelta(seconds=1),
                                             read_supermag=False)
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

    # Plot index results
    #plot_indices(sw,log,omni,outPath)
    #plot_indices(sw2,log2,omni,outPath+'/unfiled/')

    # Plot standoff distance
    #plot_standoff(mp,sw,sw2,outPath)

    # Plot energy flux
    #plot_energy_flux(mp,sw2,omni,log2,outPath)

    # Plot sat data
    #plot_satellites(themisA,themisD,themisE,
    #                old_themisA,old_themisD,old_themisE,
    #                outPath)

    # Investigate energy input for saturation
    plot_saturation(mp,dataset,outPath)
