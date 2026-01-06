#!/usr/bin/env python3
"""Final analysis and plots for the Gannon storm GRL paper
"""
import os,sys,glob,time
import numpy as np
from numpy import sin,cos,deg2rad,rad2deg,pi
import scipy
import statsmodels.api as sm
import datetime as dt
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from cmcrameri import cm as cm2
from tqdm import tqdm
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
from global_energetics.extract.magnetometer import(read_MGL,loadmagnetometers)
#
from global_energetics.wind_to_swmfInput import (collect_themis,collect_mms,
                                             collect_cluster,collect_geotail,
                                             collect_goes)
#
from supermag_api import (SuperMAGGetInventory,SuperMAGGetData,
                          SuperMAGGetIndices)

def K1label() -> str:
    return r'$\overline{\int_O{\mathbf{K}\cdot\mathbf{n}}}$'


def plot_colorline(x,y,c,ax):
    #cmax = max(abs(np.min(c)),abs(np.max(c)))
    #col = cm.twilight_shifted((c-cmax)/(2*cmax))
    col = cm.twilight_shifted_r((c-np.min(c))/(np.max(c)-np.min(c)))
    #ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.fill_between([x[i],x[i+1]], [y[i],y[i+1]], fc=col[i])
    im = ax.scatter(x, y, c=c, s=0, cmap=cm.twilight_shifted_r)
    return im

def proc_supermag(station:str,start:dt.datetime,
                                end:dt.datetime) -> pd.DataFrame:
    t0 = dt.datetime(1970,1,1)
    hardcopy = f"../data/supermag/{station}_hardcopy.csv"
    if os.path.exists(hardcopy):
        print(f"{hardcopy} found ...")
        sm_data = pd.read_csv(hardcopy,index_col='time')
        sm_data.index = pd.to_datetime(sm_data.index)
        sm_data.index.name = 'time'
    else:
        print(f'{hardcopy} not found')
        print(f'Calling supermag for {start} - {end} ...')
        userid = 'aubr'
        duration = int((end-start).total_seconds())
        status,sm_data = SuperMAGGetData(userid,start.isoformat(),duration,
                                                                   '',station)
        # Adjust the columns to be easier to work with
        sm_data['time'] = [t0+dt.timedelta(seconds=t) for t in sm_data['tval']]
        sm_data.index = sm_data['time']
        sm_data['dBn'] = [v['nez'] for v in sm_data['N']]
        sm_data['dBe'] = [v['nez'] for v in sm_data['E']]
        sm_data['dBd'] = [v['nez'] for v in sm_data['Z']]
        sm_data.drop(columns=['tval','time','N','E','Z'],inplace=True)
        sm_data.to_csv(hardcopy)
    return sm_data

def draw_imf_panel(ax:plt.Axes,solarwind:pd.DataFrame,
                                        **kwargs:dict) -> plt.Axes:
    ax.fill_between(solarwind.index,
                    np.sqrt(solarwind['bx']**2+
                            solarwind['by']**2+solarwind['bz']**2),
                    label='|B|',fc='grey')
    ax.plot(solarwind.index,solarwind['bx'],label='Bx',c='goldenrod',lw=3)
    ax.plot(solarwind.index,solarwind['by'],label='By',c='purple',lw=3)
    ax.plot(solarwind.index,solarwind['bz'],label='Bz',c='deepskyblue',lw=4)
    return ax

def draw_plasma_panel(ax:plt.Axes,solarwind:pd.DataFrame,mp:pd.DataFrame,
                                           **kwargs:dict) -> plt.Axes:
    ax.fill_between(mp.index,mp['X_subsolar [Re]'],label='SWMF',fc='grey')
    ax.plot(solarwind.index,solarwind['r_shue98'],label='Shue',
            c='black',ls='--',lw=3)
    rax = ax.twinx()
    rax.plot(solarwind.index,solarwind['density'],
             label=r'n $\left[\#/cc\right]$',c='goldenrod',lw=4)
    rax.plot(solarwind.index,solarwind['pdyn'],
             label=r'$P_{dyn}\left[nPa\right]$',c='red')
    rax.plot(solarwind.index,solarwind['Ma'],label=r'$M_A$',
             c='blue',lw=3)

    rax.set_ylabel(r'$n$ / $P_{dyn}$ / $M_A$')
    rax.legend(loc='upper right')
    rax.spines['right'].set_color('red')
    rax.tick_params(axis='y',colors='blue')

    return ax

def draw_Esw_panel(ax:plt.Axes,
            solarwind:pd.DataFrame,
                   mp:pd.DataFrame,ie:dict,**kwargs:dict) -> plt.Axes:
    # MP input energy flux
    K1     = (mp['K_netK1 [W]']+mp['UtotM1 [W]'])/-1e12
    K1_ave = (mp['K_netK1 [W]']+mp['UtotM1 [W]']).rolling('600s').mean()/-1e12

    # Joule Heating
    JH_N = np.sum(ie['N']['JouleHeat [mW/m^2]']*
                  ie['N']['Area [Re^2]']*6.371**2*1e-3,axis=1)
    JH_S = np.sum(ie['S']['JouleHeat [mW/m^2]']*
                  ie['S']['Area [Re^2]']*6.371**2*1e-3,axis=1)
    #TODO - hemispheric power doesnt make sense... NOTE skipping fo now
    HP_N = np.sum(ie['N']['E-Flux [W/m^2]']*ie['N']['Area [Re^2]']*6.371**2,
                  axis=1)
    HP_S = np.sum(ie['S']['E-Flux [W/m^2]']*ie['S']['Area [Re^2]']*6.371**2,
                  axis=1)

    ax.plot(solarwind.index,solarwind['Esw']/1e3,label=r'$E_{KL}$',
            c='black',lw=4)
    ax.fill_between(solarwind.index,solarwind['EinWang']/1e12,
                    label='$E_{in}$',fc='grey')
    ax.plot(mp.index,K1_ave,c='magenta',label=K1label())
    ax.plot(ie['N']['time'],JH_N+JH_S,c='goldenrod',label='Joule Heating',
            lw=5)
    return ax

def draw_dst_panel(ax:plt.Axes,swmf_log:pd.DataFrame,
                   omni:pd.DataFrame,**kwargs:dict) -> plt.Axes:
    ax.plot(omni.index,omni['sym_h'],label='OMNI',c='black',lw=4)
    ax.plot(swmf_log.index,swmf_log['dst_sm'],label='SWMF',c='magenta',lw=3)
    return ax

def plot_figure_1(path:str,solarwind:pd.DataFrame,
                            swmf_log:pd.DataFrame,
                                  mp:pd.DataFrame,
                                  ie:dict,
                                omni:pd.DataFrame,**kwargs:dict) -> plt.Axes:
    # Figure
    fig,axes = plt.subplots(4,figsize=[24,32],sharex=True)
    # Plots
    for ax in axes[2:4]:
        ax.axvspan(TINIT,TMAIN,fc='red',alpha=0.1)
        ax.axvspan(TMAIN,TEND,fc='blue',alpha=0.1)
    axes[0] = draw_imf_panel(axes[0],solarwind)
    axes[1] = draw_plasma_panel(axes[1],solarwind,mp)
    axes[2] = draw_Esw_panel(axes[2],solarwind,mp,ie)
    axes[3] = draw_dst_panel(axes[3],swmf_log,omni)
    # Decorate
    general_plot_settings(axes[0],do_xlabel=False,legend=True,
                          ylabel=r'IMF $\left[nT\right]$',
                          xlim=[TINIT,TEND],timedelta=False)
    general_plot_settings(axes[1],do_xlabel=False,legend=True,
                          legend_loc='upper left',
                          ylabel=r'$X \left[R_e\right]$ ',
                          xlim=[TINIT,TEND],ylim=[3,12],
                          timedelta=False)
    general_plot_settings(axes[2],do_xlabel=False,legend=True,
            ylabel=r'$E_{KL}\left[mV/m\right]$   /   $E_{in}\left[TW\right]$',
                          xlim=[TINIT,TEND],ylim=[0,82],
                          timedelta=False)
    general_plot_settings(axes[3],do_xlabel=True,legend=True,
                          ylabel=r'SYM-H $\left[nT\right]$',
                          xlim=[TINIT,TEND],timedelta=False)
    axes[-1].set_xlabel('Time [dy-hr]')
    axes[0].text(0.01,0.02,f"(a)",transform=axes[0].transAxes,
                  c='black',horizontalalignment='left',fontsize=36)
    axes[1].text(0.15,0.93,f"(b)",transform=axes[1].transAxes,
                  c='black',horizontalalignment='left',fontsize=36)
    axes[2].text(0.02,0.93,f"(c)",transform=axes[2].transAxes,
                  c='black',horizontalalignment='left',fontsize=36)
    axes[3].text(0.01,0.02,f"(d)",transform=axes[3].transAxes,
                  c='black',horizontalalignment='left',fontsize=36)

    for ax in axes:
        ax.margins(x=0.01)
        ax.grid()
    fig.tight_layout(pad=1)

    # Save
    figurename = f"{path}/figure1.png"
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)

#############################################################################

def draw_vsat_panel(ax:plt.Axes,sats:pd.DataFrame,
                    vsats:pd.DataFrame,**kwargs:dict) -> plt.Axes:
    goes  =  sats['goes16']
    vgoes = vsats['goes16']
    rax = ax.twinx()
    rax.plot(vgoes.index,vgoes['theta1'],label=r'Foot Lat.',c='grey',ls='--')

    rax.set_ylabel(r'$\theta_{I}\left[^\circ\right]$')
    rax.set_ylim([30,75])
    rax.legend(loc='lower right')
    rax.spines['right'].set_color('grey')
    rax.tick_params(axis='y',colors='grey')

    ax.plot(goes.index,goes['bz_gsm'],c='black',lw=2,label='GOES16')
    ax.plot(vgoes.index,vgoes['Bz'],c='deepskyblue',lw=4,label='SWMF')
    ax.axhline(0,c='grey',lw=4)
    # add vertical lines for when the spacecraft is on the dayside
    clock = 12+np.arctan2(vgoes['Y'],vgoes['X'])*12/np.pi
    nine = abs(clock-9)<0.001
    fifteen = abs(clock-15)<0.001
    ax.axvline(clock[nine].index[0],c='goldenrod',lw=3)
    ax.axvline(clock[fifteen].index[0],c='goldenrod',lw=3)
    ax.text(clock[nine].index[0],220,f"09",c='goldenrod',fontsize=24,ha='right')
    ax.text(clock[fifteen].index[0],220,f"15",c='goldenrod',fontsize=24,
            ha='right')
    return ax

def draw_magnetometer_panel(ax:plt.Axes,vmagnets:pd.DataFrame,
                                                  station:str) -> plt.Axes:
    # Call supermag to get the obs data
    #   NOTE pass by reference here, supermag is a global dict
    supermag = proc_supermag(station,vmagnets.index[0],vmagnets.index[-1])
    swmf_single = vmagnets[vmagnets['IAGA']==station]
    # Draw both lines on this axis with some settings
    ax.fill_between(swmf_single.index,swmf_single['dBn'].values,
                    fc='deepskyblue')
    ax.plot(supermag.index,supermag['dBn'],label='dBn_sm',c='black',lw=3)
    #plot_colorline(swmf_single.index,swmf_single['dBn'].values,
    #               swmf_single['mlt'].values,ax)
    ax.plot(swmf_single.index,swmf_single['dBn'].values,c='grey',label='_no')
    ax.axvline(swmf_single.index[abs(swmf_single['mlt']-9)<0.1][0],
               c='goldenrod',lw=3)
    ax.axvline(swmf_single.index[abs(swmf_single['mlt']-15)<0.1][0],
               c='goldenrod',lw=3)
    ax.text(swmf_single.index[abs(swmf_single['mlt']-9)<0.1][0],1250,
            f"09",c='goldenrod',horizontalalignment='right',
            fontsize=24)
    ax.text(swmf_single.index[abs(swmf_single['mlt']-15)<0.1][0],1250,
            f"15",c='goldenrod',horizontalalignment='right',
            fontsize=24)
    return ax

def draw_ampere_panel(ax:plt.Axes,infile:str,**kwargs:dict) -> plt.Axes:
    return ax

def draw_polarcap_panel(ax:plt.Axes,infile:str,**kwargs:dict) -> plt.Axes:
    return ax

def draw_FAC_panel(ax:plt.Axes,I_ampere:pd.DataFrame,
               I_swmf:pd.DataFrame,swipe:dict,**kwargs:dict) -> plt.Axes:
    I_ampere = I_ampere.sort_index()
    # Get correlation coefficient for model/data
    t_swmf = [float(t.to_numpy()) for t in I_swmf.index-TMIN]
    t_ampere = [float(t.to_numpy()) for t in I_ampere.index-TMIN]
    t_swipe = [(t-TMIN).total_seconds()*1e9 for t in swipe['time']]

    x_north = np.interp(t_swmf,t_ampere,
                        I_ampere['I_total_up_North_[MA]'].values)
    x2_north = np.interp(t_swipe,t_ampere,
                        I_ampere['I_total_up_North_[MA]'].values)

    X_north = np.column_stack((x_north,np.ones(len(x_north))))
    X2_north = np.column_stack((x2_north,np.ones(len(x2_north))))

    y_north = I_swmf['up_north_MA'].values
    y2_north = swipe['I_up_N']

    model_north = sm.OLS(y_north,X_north)
    model2_north = sm.OLS(y2_north,X2_north)

    result_north = model_north.fit()
    result2_north = model2_north.fit()

    # Plot
    ax.plot(I_ampere.index,I_ampere['I_total_up_North_[MA]'],
             label='AMPERE',c='black',lw=3)
    ax.plot(I_swmf.index,I_swmf['up_north_MA'],label='SWMF',
             c='magenta')
    ax.plot(swipe['time'],swipe['I_up_N'],label='AMPS',c='orange')

    ax.text(0.99,0.90,r'$R^2$'+f'={result_north.rsquared:.2f}',
             transform=ax.transAxes,c='magenta',horizontalalignment='right')
    ax.text(0.99,0.80,r'$R^2$'+f'={result2_north.rsquared:.2f}',
             transform=ax.transAxes,c='orange',horizontalalignment='right')
    return ax

def draw_CPCP_panel(ax:plt.Axes,pc:pd.DataFrame,
                    swmf_log:pd.DataFrame,**kwargs:dict) -> plt.Axes:
    #ax.plot(pc.index,pc['cpcpn'],label='Ridley&Kihn',c='black',lw=3)
    #ax.plot(pc.index,pc['cpcps'],label='_Ridley&Kihn',c='black',lw=1.5,
    #                                                                  ls='--')
    ax.plot(swmf_log.index,swmf_log['cpcpn'],label='SWMF_N',c='magenta',lw=3)
    ax.plot(swmf_log.index,swmf_log['cpcps'],label='_SWMF_S',c='magenta',
            lw=1.5,ls='--')
    return ax

def draw_swipe_north_tseries(axis:plt.Axes,swipe:dict,**kwargs:dict) -> None:
    axis.plot(swipe['time'],swipe['cpcp_n'],label='SWIPE',color='orange')
    return

def draw_swipe_south_tseries(axis:plt.Axes,swipe:dict,**kwargs:dict) -> None:
    axis.plot(swipe['time'],swipe['cpcp_s'],label='SWIPE',color='orange')
    return

def draw_dmsp_north_tseries(axis:plt.Axes,dmsp:dict,**kwargs:dict) -> None:
    axis.plot(dmsp['F16_N']['time'],dmsp['F16_N']['cpcp_kV'],label='DMSP F16',
              color='red')
    axis.plot(dmsp['F17_N']['time'],dmsp['F17_N']['cpcp_kV'],label='DMSP F17',
              color='blue')
    axis.plot(dmsp['F18_N']['time'],dmsp['F18_N']['cpcp_kV'],label='DMSP F18',
              color='black')
    axis.scatter(dmsp['F16_N']['time'],dmsp['F16_N']['cpcp_kV'],label='_F16_N',
              color='red',s=150,marker='X')
    axis.scatter(dmsp['F17_N']['time'],dmsp['F17_N']['cpcp_kV'],label='_F17_N',
              color='blue',s=150,marker='X')
    axis.scatter(dmsp['F18_N']['time'],dmsp['F18_N']['cpcp_kV'],label='_F18_N',
              color='black',s=150,marker='X')
    return

def draw_dmsp_south_tseries(axis:plt.Axes,dmsp:dict,**kwargs:dict) -> None:
    axis.plot(dmsp['F16_S']['time'],dmsp['F16_S']['cpcp_kV'],label='DMSP F16',
              color='red')
    axis.plot(dmsp['F17_S']['time'],dmsp['F17_S']['cpcp_kV'],label='DMSP F17',
              color='blue')
    axis.plot(dmsp['F18_S']['time'],dmsp['F18_S']['cpcp_kV'],label='DMSP F18',
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
    axis.fill_between(swmf['time'],cpcp,label='SWMF CPCP',
                      ec='grey',fc='lightgrey')
    axis.plot(dmsp['F16_N']['time'],dmsp['F16_N']['ie_cpcp'],
                 label='SWMF F16',color='orange',ls='--',lw=4)
    axis.plot(dmsp['F17_N']['time'],dmsp['F17_N']['ie_cpcp'],
                 label='SWMF F17',color='purple',ls='--',lw=4)
    axis.plot(dmsp['F18_N']['time'],dmsp['F18_N']['ie_cpcp'],
                 label='SWMF F18',color='dimgrey',ls='--',lw=4)
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
                 label='SWMF F16',color='orange',ls='--')
    axis.plot(dmsp['F17_S']['time'],dmsp['F17_S']['ie_cpcp'],
                 label='SWMF F17',color='purple',ls='--')
    axis.plot(dmsp['F18_S']['time'],dmsp['F18_S']['ie_cpcp'],
                 label='SWMF F18',color='dimgrey',ls='--')
    axis.scatter(dmsp['F16_S']['time'],dmsp['F16_S']['ie_cpcp'],
                 label='_swmfF16',color='orange',s=150,marker='o')
    axis.scatter(dmsp['F17_S']['time'],dmsp['F17_S']['ie_cpcp'],
                 label='_swmfF17',color='purple',s=150,marker='o')
    axis.scatter(dmsp['F18_S']['time'],dmsp['F18_S']['ie_cpcp'],
                 label='_swmfF18',color='dimgrey',s=150,marker='o')
    return

def dual_half_circle(center:[float,float],
                     radius:float,
                      angle:int=90,
                         ax:plt.Axes=None,
                     colors:[str,str]=('black','white'),
                   **kwargs:dict) -> [patches.Wedge,patches.Wedge]:
    """
    Add two half circles to the axes *ax* (or the current axes) with the
    specified facecolors *colors* rotated at *angle* (in degrees).
    """
    if ax is None:
        ax = plt.gca()
    theta1, theta2 = angle, angle + 180
    w1 = patches.Wedge(center, radius, theta1, theta2, ec=colors[0],
                       fc=colors[0], **kwargs)
    w2 = patches.Wedge(center, radius, theta2, theta1, ec=colors[0],
                       fc=colors[1], **kwargs)
    for wedge in [w1, w2]:
        ax.add_artist(wedge)
    return [w1, w2]

def draw_orbits(axis:plt.Axes,
                sats:dict,vsats:dict,
           solarwind:pd.DataFrame) -> None:
    # Get the Shue magnetopause at its most compressed
    sw_min = solarwind.iloc[solarwind['r_shue98'].argmin()]
    sw_max = solarwind.iloc[solarwind['r_shue98'].argmax()]
    zenith = np.linspace(160,0,100)*np.pi/180
    r_shue_min = sw_min['r_shue98']*(2/(1+cos(zenith)))**sw_min['alpha']
    X_shue_min = r_shue_min*cos(zenith)
    Y_shue_min = r_shue_min*sin(zenith)
    r_shue_max = sw_max['r_shue98']*(2/(1+cos(zenith)))**sw_max['alpha']
    X_shue_max = r_shue_max*cos(zenith)
    Y_shue_max = r_shue_max*sin(zenith)
    Y_low = np.interp(X_shue_max,X_shue_min,Y_shue_min)
    # Get the portions of the orbits that are shown in other axes
    goes = vsats['goes16'][(vsats['goes16'].index>TINIT)&
                           (vsats['goes16'].index<TEND)]
    themis = sats['themisB'][(sats['themisB'].index>TINIT)&
                             (sats['themisB'].index<TEND)]
    # highlight negative Bz as red
    sheath = goes['Bz']<0
    # Draw
    axis.fill_between(X_shue_max,Y_low,Y_shue_max,fc='grey',alpha=0.6)
    axis.fill_between(X_shue_max,-Y_low,-Y_shue_max,fc='grey',alpha=0.6)
    axis.scatter(goes['X'],goes['Y'],c='deepskyblue')
    axis.scatter(goes['X'][sheath],goes['Y'][sheath],c='red')
    axis.scatter(themis['x_gsm'],themis['y_gsm'],c='orange')
    axis.axvline(32,c='black',lw=1.5)
    axis.text(30,-15,'SWMF\nUpstream',c='black',fontsize=18)
    axis.text(15,15,'GOES',c='deepskyblue',fontsize=18)
    axis.text(15,20,'(Bz<0)',c='red',fontsize=18)
    axis.text(50,50,'THEMIS B',c='orange',fontsize=18)
    axis.text(-10,55,'(b)',c='black',fontsize=36)
    # Add Earth
    dual_half_circle((0,0),1,ax=axis)
    # Simple decorations
    axis.set_xlim(55,-10)
    axis.set_ylim(55,-25)
    axis.grid()
    return

def plot_figure_2(path:str,mp:pd.DataFrame,
                    solarwind:pd.DataFrame,
                           sats:pd.DataFrame,
                          vsats:pd.DataFrame,
                       vmagnets:pd.DataFrame,
                       I_ampere:pd.DataFrame,
                         I_swmf:pd.DataFrame,
                             pc:pd.DataFrame,
                       swmf_log:pd.DataFrame,
                           dmsp:dict,ie:dict,
                          swipe:dict,**kwargs:dict) -> plt.Axes:
    t_sw = [float(t.to_numpy()) for t in solarwind.index-TMIN]
    t_mp = [float(t.to_numpy()) for t in mp.index-TMIN]
    v = np.interp(t_mp,t_sw,solarwind['v'])
    B = np.interp(t_mp,t_sw,solarwind['B'])
    Ma = np.interp(t_mp,t_sw,solarwind['Ma'])
    clock = np.interp(t_mp,t_sw,np.arccos(solarwind['bz']/solarwind['B']))
    Rms = mp['X_subsolar [Re]']
    R05 = (1e-4*v**2 + 11.7*B*(1-np.exp(-Ma/3))*abs(np.sin(clock/2)**3))*Rms/9

    stations = ['FMC','MEA','T43']
    station_titles = [r'FMC $\left(64.3^{\circ}\right)$',
                      r'MEA $\left(61.9^{\circ}\right)$',
                      r'T43 $\left(57.9^{\circ}\right)$']
    ampere_path = '../data/ampere/'
    #ampere_quicklook='1715372880.north.png'
    ampere_quicklook='1715372880.north_annotated.png'
    ampere_image = plt.imread(f"{ampere_path}{ampere_quicklook}")
    paraview_path = '../outputs/vis/'
    #paraview_compare = 'FAC_ampere_compare.png'
    paraview_compare = 'FAC_ampere_compare_annotated.png'
    paraview_image = plt.imread(f"{paraview_path}{paraview_compare}")
    # Figure
    fig = plt.figure(figsize=[24,32])
    # GridSpecs #TODO reduce whitespace
    fivepiece = plt.GridSpec(4,1,hspace=0.1,figure=fig,
                             left=0.09,right=0.92,bottom=0.04,top=0.98,
                             height_ratios=[1,2,0.7,1.3])
    middleSection = fivepiece[1].subgridspec(1,2,hspace=0.1,wspace=0.01,
                                             width_ratios=[3,1])
    satPanels = fivepiece[0].subgridspec(1,2,wspace=0.15,width_ratios=[3,1])
    magPanels = middleSection[0].subgridspec(3,1,hspace=0.05)
    polarPanels = middleSection[1].subgridspec(2,1,hspace=0.01)
    # Declare axes
    sat_ax = fig.add_subplot(satPanels[0])
    orbit_ax = fig.add_subplot(satPanels[1])
    mag_ax1 = fig.add_subplot(magPanels[0])
    mag_ax2 = fig.add_subplot(magPanels[1])
    mag_ax3 = fig.add_subplot(magPanels[2])
    pole_ax1 = fig.add_subplot(polarPanels[0])
    pole_ax2 = fig.add_subplot(polarPanels[1])
    fac_ax = fig.add_subplot(fivepiece[2])
    cpcp_ax = fig.add_subplot(fivepiece[3])
    # Plot
    sat_ax  = draw_vsat_panel(sat_ax,sats,vsats)
    draw_orbits(orbit_ax,sats,vsats,solarwind)
    mag_ax1 = draw_magnetometer_panel(mag_ax1,vmagnets,stations[0])
    mag_ax2 = draw_magnetometer_panel(mag_ax2,vmagnets,stations[1])
    mag_ax3 = draw_magnetometer_panel(mag_ax3,vmagnets,stations[2])
    pole_ax1.imshow(ampere_image)
    pole_ax2.imshow(paraview_image)
    fac_ax  = draw_FAC_panel(fac_ax,I_ampere,I_swmf,swipe)
    draw_swmf_north_tseries(cpcp_ax,dmsp,ie['N'])
    draw_dmsp_north_tseries(cpcp_ax,dmsp)
    draw_swipe_north_tseries(cpcp_ax,swipe)
    #NOTE had this added, but the plots already waay too busy...
    #cpcp_ax.plot(R05.index,R05,c='magenta',label='Ridley 2005')

    # Decorate
    general_plot_settings(sat_ax,do_xlabel=False,legend=True,
                          legend_loc='lower left',
                          ylabel=r'$B_Z\left[nT\right]$',
                          xlim=[TINIT,TCUT-dt.timedelta(minutes=80)],
                          timedelta=False)
    sat_ax.text(0.99,0.2,f"(a)",transform=sat_ax.transAxes,
                  c='black',horizontalalignment='right',fontsize=36)
    orbit_ax.set_xlabel('X GSM [R]')
    orbit_ax.xaxis.tick_top()
    #orbit_ax.xaxis.set_label_position("top")
    orbit_ax.set_ylabel('Y GSM [R]')
    orbit_ax.yaxis.tick_right()
    orbit_ax.yaxis.set_label_position("right")
    letters = ['(c)','(d)','(e)']
    for i,ax in enumerate([mag_ax1,mag_ax2,mag_ax3]):
        general_plot_settings(ax,do_xlabel=False,legend=False,
                              ylabel=f'{station_titles[i]}',
                              ylim=[-1100,1500],
                              xlim=[TINIT,TCUT],timedelta=False)
        ax.set_xticks(sat_ax.get_xticks())
        ax.set_xlim([TINIT,TCUT])
        if i<2:
            ax.set_xticklabels([])
        ax.text(0.99,0.86,f"{letters[i]}",transform=ax.transAxes,
                c='black',horizontalalignment='right',fontsize=36)
    pole_ax1.axis('off')
    pole_ax1.text(0.98,0.98,f"05-10 20:30:00",transform=pole_ax1.transAxes,
                  c='grey',horizontalalignment='right',fontsize=24)
    pole_ax1.text(0.98,0.02,f"(f)",transform=pole_ax1.transAxes,
                  c='black',horizontalalignment='right',fontsize=36)
    pole_ax2.axis('off')
    pole_ax2.text(0.98,0.02,f"(g)",transform=pole_ax2.transAxes,
                  c='black',horizontalalignment='right',fontsize=36)
    general_plot_settings(fac_ax,do_xlabel=False,legend=True,
                          legend_loc='upper left',ylim=[0,40],
                          ylabel=r'$\int$FAC $\left[MA\right]$',
                          xlim=[TINIT,TEND], timdelta=False)
    fac_ax.set_xticklabels([])
    fac_ax.text(0.98,0.02,f"(h)",transform=fac_ax.transAxes,
                  c='black',horizontalalignment='right',fontsize=36)
    general_plot_settings(cpcp_ax,do_xlabel=True,legend=False,
                          xlim=[TINIT,TEND],ylim=[0,1150],
                          ylabel=r'CPCP $\left[kV\right]$',timedelta=False)
    cpcp_ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.60),
                   ncol=2, fancybox=True, shadow=True)
    cpcp_ax.text(0.98,0.02,f"(i)",transform=cpcp_ax.transAxes,
                  c='black',horizontalalignment='right',fontsize=36)
    cpcp_ax.set_xlabel('Time [dy-hr]')
    for ax in [sat_ax,mag_ax1,mag_ax2,mag_ax3,fac_ax,cpcp_ax]:
        ax.margins(x=0.01)
        ax.grid()

    # Save
    figurename = f"{path}/figure2.png"
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)

#############################################################################

def draw_scatter_panel(ax:plt.Axes,
                        X:pd.DataFrame,
                        Y:pd.DataFrame,
               scat_color:str, **kwargs:dict) ->[plt.Axes,plt.scatter]:
    tpre   = X.index<TMAIN
    tstorm = X.index>TMAIN
    trecovery = X.index>TMIN
    team_colors = ['red','blue']
    markers = ['o','o']
    nbins = [11,31]
    for i,phase in enumerate([tpre,tstorm]):
        # Get Pearson r
        slope,intercept,r,p,std_err = scipy.stats.linregress(X[phase].values,
                                                             Y[phase].values)
        # Obtain low,50, and high %tiles, and variance binned by our X axis
        X_bins   = np.linspace(X[phase].quantile(0.005),
                               X[phase].quantile(0.995),nbins[i])
        bin_Ranges = bin_and_describe(X[phase],Y[phase],Y[phase],X_bins,
                                      0.05,0.95)
        # Plot
        if kwargs.get('text_loc','left')=='left':
            ax.text(0.02,0.94-0.06*i,r'$R^2$'+f'={r**2:.2f}',
                    transform=ax.transAxes,
                    c=team_colors[i],horizontalalignment='left')
        elif kwargs.get('text_loc','left')=='right':
            ax.text(0.98,0.94-0.06*i,r'$R^2$'+f'={r**2:.2f}',
                    transform=ax.transAxes,
                    c=team_colors[i],horizontalalignment='right')
        #extended_fill_between(ax,X_bins,bin_Ranges['pLow_all'],
        #                                bin_Ranges['pHigh_all'],'gold',0.2)
        sc = ax.scatter(X[phase],Y[phase],marker=markers[i],
                        c=scat_color[phase],ec=team_colors[i],
                        #vmin=0,vmax=40,
                            s=50,alpha=0.8,cmap='Grays')
        ax.plot(X_bins,bin_Ranges['p50_all'],c=team_colors[i],lw=4)
        ax.plot(X_bins,slope*X_bins+intercept,c=team_colors[i],ls='--',lw=3)
    # Get Pearson r for all data
    slope,intercept,r,p,std_err = scipy.stats.linregress(X.values,
                                                         Y.values)
    if kwargs.get('text_loc','left')=='left':
        ax.text(0.02,0.94-0.06*(i+1),r'$R^2$'+f'={r**2:.2f}',
                transform=ax.transAxes,c='grey',horizontalalignment='left')
    elif kwargs.get('text_loc','left')=='right':
        ax.text(0.98,0.94-0.06*(i+1),r'$R^2$'+f'={r**2:.2f}',
                transform=ax.transAxes,c='grey',horizontalalignment='right')
    X_bins   = np.linspace(X.quantile(0.005),X.quantile(0.995),33)
    ax.plot(X_bins,slope*X_bins+intercept,c='grey',ls='--',lw=3)
    return ax,sc

def draw_fit_resid_panel(ax:plt.Axes,
                            x:np.ndarray,
                            y:np.ndarray,
                   scat_color:np.ndarray,
                           **kwargs:dict)->[plt.Axes,plt.scatter]:
    tpre   = x.index<TMAIN
    tstorm = x.index>TMAIN
    team_colors = ['red','blue']
    markers = ['o','o']
    for i,phase in enumerate([tpre,tstorm]):
        xp = x[phase]
        yp = y[phase]
        colors = scat_color[phase]
        x_sort = xp.argsort()
        xp = xp[x_sort]
        # Create a model for prediction using Ordinary Least Squares regression
        if kwargs.get('form','linear')=='linear':
            X = np.column_stack((xp,np.ones(len(xp))))
        elif kwargs.get('form','linear')=='sq':
            X = np.column_stack((xp,xp**(1/2),np.ones(len(xp))))
        elif kwargs.get('form','linear')=='cu':
            X = np.column_stack((xp,xp**(1/3),np.ones(len(xp))))
        yp = yp[x_sort]

        model  = sm.OLS(yp,X)
        result = model.fit()

        sc = ax.scatter(result.fittedvalues,
        #                (result.fittedvalues-yp)/yp,
                        result.fittedvalues-yp,
                        s=50,alpha=0.8,cmap='Grays',vmin=0,vmax=40,
                        c=colors[x_sort],ec=team_colors[i])
    ax.axhline(0,c='gray')
    return ax, sc

def draw_model_resid_panel(ax:plt.Axes,
                        model:np.ndarray,
                     observed:np.ndarray,
                   scat_color:np.ndarray,
                           **kwargs:dict)->[plt.Axes,plt.scatter]:
    sc = ax.scatter(model,(model-observed)/observed,s=50,alpha=0.8,
                    cmap=cm.managua,c=scat_color)
    ax.axhline(0,c='grey')
    return ax, sc

def plot_figure_3(path:str,solarwind:pd.DataFrame,
                                  mp:pd.DataFrame,
                              I_swmf:pd.DataFrame,
                            I_ampere:pd.DataFrame,
                            swmf_log:pd.DataFrame,**kwargs:dict) -> plt.Axes:

    # Get data to a common time axis
    t_sw = [float(t.to_numpy()) for t in solarwind.index-TMIN]
    t_mp = [float(t.to_numpy()) for t in mp.index-TMIN]
    t_log = [float(t.to_numpy()) for t in swmf_log.index-TMIN]
    t_ie   = [float(t.to_numpy()) for t in I_swmf.index-TMIN]
    t_ampere = [float(t.to_numpy()) for t in I_ampere.index-TMIN]
    K1  = (mp['K_netK1 [W]']+mp['UtotM1 [W]']).rolling('600s').mean()/-1e12
    Ein = pd.Series(index=K1.index,
                   data=np.interp(t_mp,t_sw,solarwind['EinWang'].values/1e12))
    Esw = pd.Series(index=K1.index,
                     data=np.interp(t_mp,t_sw,solarwind['Esw'].values/1e3))
    CPCP= pd.Series(index=K1.index,
                     data=np.interp(t_mp,t_log,swmf_log['cpcpn'].values))
    FAC  = pd.Series(index=K1.index,data=np.interp(t_mp,t_ie,
                      (I_swmf['up_north_MA']).values))
    AMP_FAC = pd.Series(index=K1.index,data=np.interp(t_mp,t_ampere,
                                    I_ampere['I_total_up_North_[MA]'].values))
    pdyn = np.interp(t_mp,t_sw,solarwind['pdyn'].values)
    times   = np.array([t/3600e9 for t in t_mp])
    BOYLE   = pd.Series(index=K1.index,
                       data=np.interp(t_mp,t_sw,solarwind['CPCP_B97'].values))
    SHILL   = pd.Series(index=K1.index,
                       data=np.interp(t_mp,t_sw,solarwind['CPCP_S02'].values))
    KRID    = pd.Series(index=K1.index,
                       data=np.interp(t_mp,t_sw,solarwind['CPCP_K08'].values))

    # Split data into two populations
    tpre   = FAC.index<TMAIN
    tstorm = FAC.index>TMAIN
    trecovery = FAC.index>TMIN
    phase = np.zeros(len(K1))
    phase[tpre] = 0
    phase[tstorm] = 8
    phase[trecovery] = 7

    K1storm = K1[tstorm]
    Einstorm = Ein[tstorm]
    Eswstorm = Esw[tstorm]
    CPCPstorm = CPCP[tstorm]
    FACstorm = FAC[tstorm]
    AMP_FACstorm = AMP_FAC[tstorm]
    pdynstorm = pdyn[tstorm]
    timesstorm = times[tstorm]
    BOYLEstorm = BOYLE[tstorm]
    SHILLstorm = SHILL[tstorm]
    KRIDstorm = KRID[tstorm]

    K1pre = K1[tpre]
    Einpre = Ein[tpre]
    Eswpre = Esw[tpre]
    CPCPpre = CPCP[tpre]
    FACpre = FAC[tpre]
    AMP_FACpre = AMP_FAC[tpre]
    pdynpre = pdyn[tpre]
    timespre = times[tpre]
    BOYLEpre = BOYLE[tpre]
    SHILLpre = SHILL[tpre]
    KRIDpre = KRID[tpre]


    # Figure
    fig = plt.figure(figsize=[30,24])
    # GridSpecs #TODO reduce whitespace
    twocolumn = plt.GridSpec(1,2,hspace=0.1,figure=fig,
                             left=0.06,right=0.95,bottom=0.07,top=0.98,
                             width_ratios=[4,56],wspace=0.1)
    fourpack = twocolumn[1].subgridspec(2,2,hspace=0.15,wspace=0.1)
    # Declare axes
    ax1 = fig.add_subplot(fourpack[0,0])
    ax2 = fig.add_subplot(fourpack[0,1])
    ax3 = fig.add_subplot(fourpack[1,0])
    ax4 = fig.add_subplot(fourpack[1,1])

    # Plot
    ax1,scl = draw_scatter_panel(ax1,Esw,K1,pdyn,text_loc='right')
    ax2,scl = draw_scatter_panel(ax2,K1[K1>0],FAC[K1>0],pdyn[K1>0])
    ax3,scl = draw_scatter_panel(ax3,FAC,CPCP,pdyn)
    ax4,scl = draw_scatter_panel(ax4,Ein,CPCP,pdyn)

    # Decorate
    cb_axl = fig.add_axes([0.06,.084,.02,.854])
    cbl = fig.colorbar(scl,orientation='vertical',cax=cb_axl)
    cbl.set_label(r"$p_{dyn}\left[nPa\right]$",fontsize=36)

    cb_axl.yaxis.set_ticks_position("left")
    cb_axl.yaxis.set_label_position("left")

    ax1.set_xlabel(r"$E_{in}\left[TW\right]$ Wang'14")
    ax1.set_ylabel(K1label())

    ax2.set_xlabel(K1label())
    ax2.set_ylabel(r'$\int$FAC $\left[MA\right]$')

    ax3.set_xlabel(r'$\int$FAC $\left[MA\right]$')
    ax3.set_ylabel(r'CPCP $\left[kV\right]$')

    ax4.set_xlabel(r"$E_{in}\left[TW\right]$ Wang'14")
    ax4.set_ylabel(r'CPCP $\left[kV\right]$')

    letters = ['(a)','(b)','(c)','(d)']
    for i,ax in enumerate([ax1,ax2,ax3,ax4]):
        if i==1 or i==3:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        ax.text(0.98,0.02,f"{letters[i]}",transform=ax.transAxes,
                c='black',horizontalalignment='right',fontsize=36)
        ax.grid()

    # Save
    figurename = f"{path}/figure3.png"
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)

##############################################################################

def plot_figure_4(path:str,solarwind:pd.DataFrame,
                                  mp:pd.DataFrame,
                              I_swmf:pd.DataFrame,
                            I_ampere:pd.DataFrame,
                            swmf_log:pd.DataFrame,**kwargs:dict) -> plt.Axes:

    # Get data to a common time axis
    t_sw = [float(t.to_numpy()) for t in solarwind.index-TMIN]
    t_mp = [float(t.to_numpy()) for t in mp.index-TMIN]
    t_log = [float(t.to_numpy()) for t in swmf_log.index-TMIN]
    t_ie   = [float(t.to_numpy()) for t in I_swmf.index-TMIN]
    t_ampere = [float(t.to_numpy()) for t in I_ampere.index-TMIN]
    K1  = (mp['K_netK1 [W]']+mp['UtotM1 [W]']).rolling('600s').mean()/-1e12
    Ein = pd.Series(index=K1.index,
                   data=np.interp(t_mp,t_sw,solarwind['EinWang'].values/1e12))
    Esw = pd.Series(index=K1.index,
                     data=np.interp(t_mp,t_sw,solarwind['Esw'].values/1e3))
    CPCP= pd.Series(index=K1.index,
                     data=np.interp(t_mp,t_log,swmf_log['cpcpn'].values))
    FAC  = pd.Series(index=K1.index,data=np.interp(t_mp,t_ie,
                      (I_swmf['up_north_MA']).values))
    AMP_FAC = pd.Series(index=K1.index,data=np.interp(t_mp,t_ampere,
                                    I_ampere['I_total_up_North_[MA]'].values))
    pdyn = np.interp(t_mp,t_sw,solarwind['pdyn'].values)
    times   = np.array([t/3600e9 for t in t_mp])
    BOYLE   = pd.Series(index=K1.index,
                       data=np.interp(t_mp,t_sw,solarwind['CPCP_B97'].values))
    SHILL   = pd.Series(index=K1.index,
                       data=np.interp(t_mp,t_sw,solarwind['CPCP_S02'].values))
    KRID    = pd.Series(index=K1.index,
                       data=np.interp(t_mp,t_sw,solarwind['CPCP_K08'].values))

    # Create some models for how the K1 FAC relationship could be fit
    x = K1[K1>0].values
    x_sort = x.argsort()
    x = x[x_sort]
    X_linear = np.column_stack((x,np.ones(len(x))))
    X_sq = np.column_stack((x,x**(1/2),np.ones(len(x))))
    X_cu = np.column_stack((x,x**(1/3),np.ones(len(x))))
    y = FAC[K1>0].iloc[x_sort]

    model_linear = sm.OLS(y,X_linear)
    model_sq = sm.OLS(y,X_sq)
    model_cu = sm.OLS(y,X_cu)

    result_linear = model_linear.fit()
    result_sq     = model_sq.fit()
    result_cu     = model_cu.fit()

    # For Ein predict CPCP
    x2      = Ein.values
    x2_sort = x2.argsort()
    x2      = x2[x2_sort]
    X2_sq   = np.column_stack((x2,x2**(1/2),np.ones(len(x2))))
    y2      = CPCP.iloc[x2_sort]

    model2_sq  = sm.OLS(y2,X2_sq)
    result2_sq = model2_sq.fit()

    # For Esw predict CPCP
    x3      = Esw.values
    x3_sort = x3.argsort()
    x3      = x3[x3_sort]
    X3      = np.column_stack((x3,x3**(1/2),np.ones(len(x3))))
    y3      = CPCP.iloc[x3_sort]

    model3  = sm.OLS(y3,X3)
    result3 = model3.fit()

    # Figure
    fig = plt.figure(figsize=[40,30])
    # GridSpecs #TODO reduce whitespace
    slivers = plt.GridSpec(1,2,hspace=0.1,figure=fig,
                             left=0.10,right=0.90,bottom=0.04,top=0.98,
                             width_ratios=[28,28],wspace=0.25)
    fits6 = slivers[0].subgridspec(3,2,hspace=0.2,wspace=0.1)
    model6 = slivers[1].subgridspec(3,2,hspace=0.2,wspace=0.1)
    # Declare axes
    fit_ax1 = fig.add_subplot(fits6[0])
    fit_ax2 = fig.add_subplot(fits6[1])
    fit_ax3 = fig.add_subplot(fits6[2])
    fit_ax4 = fig.add_subplot(fits6[3])
    fit_ax5 = fig.add_subplot(fits6[4])
    fit_ax6 = fig.add_subplot(fits6[5])

    model_ax1 = fig.add_subplot(model6[0])
    model_ax2 = fig.add_subplot(model6[1])
    model_ax3 = fig.add_subplot(model6[2])
    model_ax4 = fig.add_subplot(model6[3])
    model_ax5 = fig.add_subplot(model6[4])
    model_ax6 = fig.add_subplot(model6[5])

    # Plot
    fit_ax1,scl = draw_scatter_panel(fit_ax1,Esw,CPCP,pdyn,text_loc='right')
    fit_ax3,scl = draw_scatter_panel(fit_ax3,K1[K1>0]**0.5,CPCP[K1>0],
                                     pdyn[K1>0])
    fit_ax5,scl = draw_scatter_panel(fit_ax5,AMP_FAC,CPCP,pdyn)

    model_ax1,scl = draw_scatter_panel(model_ax1,BOYLE,CPCP,pdyn,
                                      text_loc='right')
    model_ax3,scl = draw_scatter_panel(model_ax3,SHILL,CPCP,pdyn)
    model_ax5,scl = draw_scatter_panel(model_ax5,KRID,CPCP,pdyn,
                                       text_loc='right')

    fit_ax2,scr = draw_fit_resid_panel(fit_ax2,Esw.values,CPCP.values,times)
    fit_ax4,scr= draw_fit_resid_panel(fit_ax4,K1[K1>0],CPCP[K1>0],
                                        times[K1>0],form='sq')
    fit_ax6,scr=draw_fit_resid_panel(fit_ax6,AMP_FAC.values,CPCP.values,times)

    model_ax2,scr= draw_model_resid_panel(model_ax2,BOYLE.values,CPCP.values,
                                          times)
    model_ax4,scr= draw_model_resid_panel(model_ax4,SHILL.values,CPCP.values,
                                          times)
    model_ax6,scr= draw_model_resid_panel(model_ax6,KRID.values,CPCP.values,
                                          times)

    # Decorate
    cb_axl = fig.add_axes([0.02,.084,.02,.854])
    cb_axr = fig.add_axes([0.94,.084,.02,.854])
    cbl = fig.colorbar(scl,orientation='vertical',cax=cb_axl)
    cbr = fig.colorbar(scr,orientation='vertical',cax=cb_axr)
    cbl.set_label(r"$p_{dyn}\left[nPa\right]$",fontsize=36)
    cbr.set_label(r"$T-$ 11-01:30:00 $\left[Hr\right]$",fontsize=36)

    cb_axl.yaxis.set_ticks_position("left")
    cb_axl.yaxis.set_label_position("left")

    fit_ax1.set_xlabel(r'$E_{KL}\left[mA/m\right]$')
    fit_ax1.plot(x3,result3.fittedvalues,c='red',ls='--')
    fit_ax1.text(0.98,0.88,r'$R^2$'+f'={result3.rsquared:.2f}',
             transform=fit_ax1.transAxes,c='red',horizontalalignment='right')
    fit_ax3.set_xlabel(
                    r"$\sqrt{\overline{\int_O{\mathbf{K}\cdot\mathbf{n}}}}$")
    fit_ax5.set_xlabel(r'AMPERE $\int$FAC $\left[MA\right]$')

    fit_ax2.set_ylabel(r"$E_{KL}$ - SWMF")
    fit_ax4.set_ylabel(
            r"$\sqrt{\overline{\int_O{\mathbf{K}\cdot\mathbf{n}}}}$ - SWMF")
    fit_ax6.set_ylabel(r"AMPERE $FAC$ - SWMF")
    fit_ax6.set_xlabel(r"Predicted CPCP $\left[kV\right]$")

    model_ax1.set_xlabel(r"Boyle'97 CPCP $\left[kV\right]$")
    model_ax3.set_xlabel(r"Siscoe-Hill'02 CPCP $\left[kV\right]$")
    model_ax5.set_xlabel(r"Kivelson-Ridley'08 CPCP $\left[kV\right]$")

    model_ax2.set_ylabel(r"Boyle'97 - SWMF")
    model_ax4.set_ylabel(r"Siscoe-Hill'02 - SWMF")
    model_ax6.set_ylabel(r"Kivelson-Ridley'08 - SWMF")
    model_ax6.set_xlabel(r"Predicted CPCP $\left[kV\right]$")


    letters = ['(a)','(d)','(b)','(e)','(c)','(f)']
    for i,ax in enumerate([fit_ax1,fit_ax2,fit_ax3,
                           fit_ax4,fit_ax5,fit_ax6]):
        if i==1 or i==3 or i==5:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_xlim([0,800])
            ax.set_ylim([-2,6])
        else:
            ax.set_ylabel(r'CPCP $\left[kV\right]$')
        ax.text(0.98,0.02,f"{letters[i]}",transform=ax.transAxes,
                c='black',horizontalalignment='right',fontsize=36)
    letters = ['(g)','(j)','(h)','(k)','(i)','(l)']
    for i,ax in enumerate([model_ax1,model_ax2,model_ax3,
                           model_ax4,model_ax5,model_ax6]):
        if i==1 or i==3 or i==5:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_xlim([0,800])
            ax.set_ylim([-2,6])
        else:
            ax.set_ylabel(r'CPCP $\left[kV\right]$')
        ax.text(0.98,0.02,f"{letters[i]}",transform=ax.transAxes,
                c='black',horizontalalignment='right',fontsize=36)
    #fig.patches.extend([plt.Rectangle([0,0.25],0.23,1,fill=True,fc='plum',
    #               alpha=0.2,zorder=-1,transform=fig.transFigure,figure=fig)])

    # Save
    figurename = f"{path}/figure4.png"
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)
#############################################################################

def draw_general_scatter(ax:plt.Axes,Y:pd.Series,X:pd.Series,
                      color:str,label:str,**kwargs:dict)->None:
    tpre   = X.index<TMAIN
    tstorm = X.index>TMAIN
    trecovery = X.index>TMIN
    phase_name = ['Pre','Storm']
    team_colors = [kwargs.get('red_shade','red'),
                   kwargs.get('blue_shade','blue')]
    markers = ['o','o']
    nbins = [11,31]
    text_X,text_Y = kwargs.get('text_xy',[0.02,0.94])
    text_head = kwargs.get('text_head',r'$R^2$=')
    for i,phase in enumerate([tpre,tstorm]):
        # Get Pearson r
        slope,intercept,r,p,std_err = scipy.stats.linregress(X[phase].values,
                                                             Y[phase].values)
        # Obtain low,50, and high %tiles, and variance binned by our X axis
        X_bins   = np.linspace(X[phase].quantile(0.005),
                               X[phase].quantile(0.995),nbins[i])
        bin_Ranges = bin_and_describe(X[phase],Y[phase],Y[phase],X_bins,
                                      0.05,0.95)
        # Plot
        if kwargs.get('text_loc','left')=='left':
            ax.text(text_X,text_Y-0.06*i,f'{text_head}{r**2:.2f}',
                    transform=ax.transAxes,
                    c=team_colors[i],horizontalalignment='left')
        elif kwargs.get('text_loc','left')=='right':
            ax.text(0.98,0.94-0.06*i,f'{text_head}{r**2:.2f}',
                    transform=ax.transAxes,
                    c=team_colors[i],horizontalalignment='right')
        sc = ax.scatter(X[phase],Y[phase],marker=markers[i],
                        c=color,ec=team_colors[i],s=50,alpha=0.8,
                        label=f"{phase_name[i]} {label}")
        ax.plot(X_bins,bin_Ranges['p50_all'],c=team_colors[i],lw=4)
        ax.plot(X_bins,slope*X_bins+intercept,c=team_colors[i],ls='--',lw=3)
    # Get Pearson r for all data
    slope,intercept,r,p,std_err = scipy.stats.linregress(X.values,
                                                         Y.values)
    if kwargs.get('text_loc','left')=='left':
        ax.text(text_X,text_Y-0.06*(i+1),f'{text_head}{r**2:.2f}',
                transform=ax.transAxes,c='grey',horizontalalignment='left')
    elif kwargs.get('text_loc','left')=='right':
        ax.text(0.98,0.94-0.06*(i+1),f'{text_head}{r**2:.2f}',
                transform=ax.transAxes,c='grey',horizontalalignment='right')
    X_bins   = np.linspace(X.quantile(0.005),X.quantile(0.995),33)
    ax.plot(X_bins,slope*X_bins+intercept,c='grey',ls='--',lw=3)
    return

def draw_swmf_sparse(ax:plt.Axes,CPCP:pd.Series,dmsp:dict,X:pd.Series) -> None:
    t_x = np.array([float(t.to_numpy()) for t in X.index-TMIN])
    xPre   = X.index<TMAIN
    xStorm = X.index>=TMAIN
    #ax.scatter(X[xPre],CPCP[xPre],ec='red',c='white',alpha=0.8,s=100,
    #           label='Pre')
    #ax.scatter(X[xStorm],CPCP[xStorm],ec='blue',c='white',alpha=0.8,s=100,
    #           label='Storm')
    # Interpolate the plotting variable Y for each sat in dmsp
    #colors = ['green','teal','cyan','dodgerblue']
    markers = ['x','o','+']
    for i,sat in enumerate(['F16_N','F17_N','F18_N']):
        if i==0: satlabel='vDMSP Pass'
        else: satlabel='_nolabel'
        clean = np.array([not b for b in np.isnan(dmsp[sat]['cpcp_kV'])])
        times = np.array(dmsp[sat]['time'][clean])
        tstarts = np.array(dmsp[sat]['tstart'][clean])
        tends = np.array(dmsp[sat]['tend'][clean])

        pre   = (times>=X.index[0]) & (times<TMAIN)
        storm = (times>=TMAIN)      & (times<X.index[-1])

        X_sparse_pre = np.zeros([len(tstarts[pre]),3])# mean, min, max
        for i,(start,end) in enumerate(zip(tstarts[pre],tends[pre])):
            interv = (X.index>start) & (X.index<end)
            X_sparse_pre[i,0] = X[interv].mean()
            X_sparse_pre[i,1] = X[interv].mean()-X[interv].quantile(0.25)
            X_sparse_pre[i,2] = X[interv].quantile(0.75)-X[interv].mean()
            #X_sparse_pre[i,1] = X[interv].std()
            #X_sparse_pre[i,2] = X[interv].std()
        err_pre = np.array([z for z in zip(X_sparse_pre[:,1],
                                           X_sparse_pre[:,2])]).T

        X_sparse_storm = np.zeros([len(tstarts[storm]),3])# mean, min, max
        for i,(start,end) in enumerate(zip(tstarts[storm],tends[storm])):
            interv = (X.index>start) & (X.index<end)
            X_sparse_storm[i,0] = X[interv].mean()
            X_sparse_storm[i,1] = X[interv].mean()-X[interv].quantile(0.25)
            X_sparse_storm[i,2] = X[interv].quantile(0.75)-X[interv].mean()
            #X_sparse_storm[i,1] = X[interv].std()
            #X_sparse_storm[i,2] = X[interv].std()
        err_storm = np.array([z for z in zip(X_sparse_storm[:,1],
                                             X_sparse_storm[:,2])]).T

        ax.scatter(X_sparse_pre[:,0],dmsp[sat]['ie_cpcp'][clean][pre],ec='red',
                   alpha=0.8,label=satlabel+' Pre',s=200,c='red')
        ax.scatter(X_sparse_storm[:,0],dmsp[sat]['ie_cpcp'][clean][storm],
                   ec='blue',
                   label=satlabel+' Storm',s=200,c='blue',alpha=0.8)

        ax.errorbar(X_sparse_pre[:,0],dmsp[sat]['ie_cpcp'][clean][pre],
                    xerr=err_pre,fmt='none',ecolor='black')
        ax.errorbar(X_sparse_storm[:,0],dmsp[sat]['ie_cpcp'][clean][storm],
                    xerr=err_storm,fmt='none',ecolor='black')

    return

def draw_dmsp_sparse(ax:plt.Axes,dmsp:dict,X:pd.Series) -> None:
    t_x = np.array([float(t.to_numpy()) for t in X.index-TMIN])
    xPre   = X.index<TMAIN
    xStorm = X.index>=TMAIN
    # Interpolate the plotting variable Y for each sat in dmsp
    #colors = ['green','teal','cyan','dodgerblue']
    markers = ['o','o','o']
    for i,sat in enumerate(['F16_N','F17_N','F18_N']):
        if i==0: dolabel=''
        else: dolabel='_'
        times = np.array(dmsp[sat]['time'])
        tstarts = np.array(dmsp[sat]['tstart'])
        tends = np.array(dmsp[sat]['tend'])

        pre   = (times>=X.index[0]) & (times<TMAIN)
        storm = (times>=TMAIN)      & (times<X.index[-1])

        X_sparse_pre = np.zeros([len(tstarts[pre]),3])# mean, min, max
        for i,(start,end) in enumerate(zip(tstarts[pre],tends[pre])):
            interv = (X.index>start) & (X.index<end)
            X_sparse_pre[i,0] = X[interv].mean()
            X_sparse_pre[i,1] = X[interv].mean()-X[interv].quantile(0.25)
            X_sparse_pre[i,2] = X[interv].quantile(0.75)-X[interv].mean()
        err_pre = np.array([z for z in zip(X_sparse_pre[:,1],
                                           X_sparse_pre[:,2])]).T

        X_sparse_storm = np.zeros([len(tstarts[storm]),3])# mean, min, max
        for i,(start,end) in enumerate(zip(tstarts[storm],tends[storm])):
            interv = (X.index>start) & (X.index<end)
            X_sparse_storm[i,0] = X[interv].mean()
            X_sparse_storm[i,1] = X[interv].mean()-X[interv].quantile(0.25)
            X_sparse_storm[i,2] = X[interv].quantile(0.75)-X[interv].mean()
        err_storm = np.array([z for z in zip(X_sparse_storm[:,1],
                                             X_sparse_storm[:,2])]).T

        ax.errorbar(X_sparse_pre[:,0],dmsp[sat]['cpcp_kV'][pre],
                    xerr=err_pre,fmt='none',ecolor='red')
        ax.errorbar(X_sparse_storm[:,0],dmsp[sat]['cpcp_kV'][storm],
                    xerr=err_storm,fmt='none',ecolor='blue')

        ax.scatter(X_sparse_pre[:,0],dmsp[sat]['cpcp_kV'][pre],ec='red',
                   alpha=0.8,label=dolabel+'Pre',s=200,c='red')
        ax.scatter(X_sparse_storm[:,0],dmsp[sat]['cpcp_kV'][storm],ec='blue',
                   label=dolabel+'Storm',s=200,c='blue',alpha=0.8)

    return


def plot_figure_5(path:str,solarwind:pd.DataFrame,
                                  mp:pd.DataFrame,
                              I_swmf:pd.DataFrame,
                            I_ampere:pd.DataFrame,
                            swmf_log:pd.DataFrame,
                  dmsp:dict,ie:dict,swipe:dict,**kwargs:dict) -> None:
    # Get data to a common time axis
    t_sw = [float(t.to_numpy()) for t in solarwind.index-TMIN]
    t_mp = [float(t.to_numpy()) for t in mp.index-TMIN]
    t_log = [float(t.to_numpy()) for t in swmf_log.index-TMIN]
    t_ie   = [float(t.to_numpy()) for t in I_swmf.index-TMIN]
    t_ampere = [float(t.to_numpy()) for t in I_ampere.index-TMIN]
    t_swipe = [(t-TMIN).total_seconds()*1e9 for t in swipe['time']]

    K1  = (mp['K_netK1 [W]']+mp['UtotM1 [W]']).rolling('600s').mean()/-1e12
    Ein = pd.Series(index=K1.index,name='Ein',
                   data=np.interp(t_mp,t_sw,solarwind['EinWang'].values/1e12))
    Esw = pd.Series(index=K1.index,name='Esw',
                     data=np.interp(t_mp,t_sw,solarwind['Esw'].values/1e3))
    CPCP= pd.Series(index=K1.index,name='CPCP',
                     data=np.interp(t_mp,t_log,swmf_log['cpcpn'].values))
    FAC  = pd.Series(index=K1.index,name='FAC',
                     data=np.interp(t_mp,t_ie,(I_swmf['up_north_MA']).values))
    AMP_FAC = pd.Series(index=K1.index,name='AMP_FAC',
                     data=np.interp(t_mp,t_ampere,
                                    I_ampere['I_total_up_North_[MA]'].values))
    pdyn = np.interp(t_mp,t_sw,solarwind['pdyn'].values)
    BOYLE   = pd.Series(index=K1.index,name='BOYLE',
                       data=np.interp(t_mp,t_sw,solarwind['CPCP_B97'].values))
    SHILL   = pd.Series(index=K1.index,name='SHILL',
                       data=np.interp(t_mp,t_sw,solarwind['CPCP_S02'].values))
    KRID    = pd.Series(index=K1.index,name='KRID',
                       data=np.interp(t_mp,t_sw,solarwind['CPCP_K08'].values))
    SWIPE = pd.Series(index=K1.index,name='SWIPE',
                      data=np.interp(t_mp,t_swipe,swipe['cpcp_n']))
    v = np.interp(t_mp,t_sw,solarwind['v'])
    B = np.interp(t_mp,t_sw,solarwind['B'])
    Ma = np.interp(t_mp,t_sw,solarwind['Ma'])
    clock = np.interp(t_mp,t_sw,np.arccos(solarwind['bz']/solarwind['B']))
    Rms = mp['X_subsolar [Re]']
    R05 = (1e-4*v**2 + 11.7*B*(1-np.exp(-Ma/3))*abs(np.sin(clock/2)**3))*Rms/9

    tpre   = FAC.index<TMAIN
    tstorm = FAC.index>TMAIN
    dmsp_sparse = {}
    for sat in ['F16_N','F17_N','F18_N']:
        clean = np.array([not b for b in np.isnan(dmsp[sat]['cpcp_kV'])])
        ids = dmsp[sat]['pass_id'][clean]
        times = dmsp[sat]['time'][clean]
        cpcp = dmsp[sat]['cpcp_kV'][clean]
        ie_cpcp = dmsp[sat]['ie_cpcp'][clean]
        uids = np.unique(ids)
        i_start = np.zeros(len(uids))
        i_sparse = np.zeros(len(uids))
        i_end = np.zeros(len(uids))
        for i,t in enumerate(uids):
            indices = np.where(ids==t)[0]
            i_start[i] = indices[0]
            i_sparse[i] = indices[int(len(indices)/2)]
            i_end[i] = indices[-1]
        i_start = np.array([int(i) for i in sorted(i_start)])
        i_sparse = np.array([int(i) for i in sorted(i_sparse)])
        i_end = np.array([int(i) for i in sorted(i_end)])
        dmsp_sparse[sat] = {}
        dmsp_sparse[sat]['time'] = times[i_sparse]
        dmsp_sparse[sat]['tstart'] = times[i_start]
        dmsp_sparse[sat]['tend'] = times[i_end]
        dmsp_sparse[sat]['cpcp_kV'] = cpcp[i_sparse]
        dmsp_sparse[sat]['ie_cpcp'] = ie_cpcp[i_sparse]
    sparse_times = np.unique(np.concat([dmsp_sparse[sat]['time'] for sat in
                                        ['F16_N','F17_N','F18_N']]))
    t_sparse = [(t-TMIN).total_seconds()*1e9 for t in sparse_times]
    Esw_sparse = pd.Series(index=sparse_times,
                   data=np.interp(t_sparse,t_sw,solarwind['Esw'].values))
    AMP_FAC_sparse = pd.Series(index=sparse_times,
                   data=np.interp(t_sparse,t_ampere,
                               I_ampere['I_total_up_North_[MA]'].values))
    # Setup Figure
    fig = plt.figure(figsize=[30,30])
    # GridSpecs
    columns = plt.GridSpec(1,2,hspace=0.1,figure=fig,
                             left=0.10,right=0.90,bottom=0.04,top=0.98,
                             width_ratios=[1,1],wspace=0.1)
    left = columns[0].subgridspec(3,1,hspace=0.05)
    right = columns[1].subgridspec(3,1,hspace=0.05)

    # Declare axes
    ax_swmf_Esw         = fig.add_subplot(left[0])
    ax_dmsp_Esw         = fig.add_subplot(left[1])
    ax_empirical_Esw    = fig.add_subplot(left[2])
    ax_swmf_ampere      = fig.add_subplot(right[0])
    ax_dmsp_ampere      = fig.add_subplot(right[1])
    ax_empirical_ampere = fig.add_subplot(right[2])

    # Draw
    draw_general_scatter(ax_swmf_Esw,CPCP,Esw,'white','SWMF')
    draw_general_scatter(ax_swmf_ampere,CPCP,AMP_FAC,'white','SWMF')
    draw_swmf_sparse(ax_swmf_Esw,CPCP,dmsp_sparse,Esw)
    draw_swmf_sparse(ax_swmf_ampere,CPCP,dmsp_sparse,AMP_FAC)
    draw_dmsp_sparse(ax_dmsp_Esw,dmsp_sparse,Esw)
    draw_dmsp_sparse(ax_dmsp_ampere,dmsp_sparse,AMP_FAC)
    '''
    draw_general_scatter(ax_empirical_Esw,BOYLE,Esw,'black','Boyle',
                         text_head='',text_xy=[0.02,.89],
                         red_shade='darkred',blue_shade='darkblue')
    draw_general_scatter(ax_empirical_ampere,BOYLE,AMP_FAC,'black','Boyle',
                         text_head='',text_xy=[0.02,.89],
                         red_shade='darkred',blue_shade='darkblue')
    '''
    draw_general_scatter(ax_empirical_Esw,R05,Esw,'black','Ridley2005',
                         text_head='',text_xy=[0.02,.89],
                         red_shade='darkred',blue_shade='darkblue')
    draw_general_scatter(ax_empirical_ampere,R05,AMP_FAC,'black','Ridley2005',
                         text_head='',text_xy=[0.02,.89],
                         red_shade='darkred',blue_shade='darkblue')
    draw_general_scatter(ax_empirical_Esw,SWIPE,Esw,'white','SWIPE',
                         text_head='',text_xy=[0.12,.89],
                         red_shade='lightcoral',blue_shade='cornflowerblue')
    draw_general_scatter(ax_empirical_ampere,SWIPE,AMP_FAC,'white','SWIPE',
                         text_head='',text_xy=[0.12,.89],
                         red_shade='lightcoral',blue_shade='cornflowerblue')

    # Decorate
    label_tag = ['SWMF','DMSP','Empirical']
    panel_tag = ['(a)','(c)','(e)']
    for i,ax in enumerate([ax_swmf_Esw,ax_dmsp_Esw,ax_empirical_Esw]):
        ax.set_ylabel(f'{label_tag[i]} CPCP '+r'$\left[kV\right]$')
        ax.set_xlim([0,45])
        ax.set_ylim([0,900])
        ax.legend()
        ax.grid()
        ax.text(0.98,0.02,f"{panel_tag[i]}",transform=ax.transAxes,
                c='black',horizontalalignment='right',fontsize=36)

    panel_tag = ['(b)','(d)','(f)']
    for i,ax in enumerate([ax_swmf_ampere,ax_dmsp_ampere,ax_empirical_ampere]):
        ax.set_ylabel(f'{label_tag[i]} CPCP '+r'$\left[kV\right]$')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_xlim([0,38])
        ax.set_ylim([0,900])
        #ax.legend()
        ax.grid()
        ax.text(0.98,0.02,f"{panel_tag[i]}",transform=ax.transAxes,
                c='black',horizontalalignment='right',fontsize=36)

    ax_empirical_Esw.set_xlabel(r'$E_{KL} \left[ mA/m\right]$')
    ax_empirical_ampere.set_xlabel(r'AMPERE TOT. FAC $\left[ MA\right]$')

    ax_empirical_Esw.text(0.02,0.94,r'$\circ R^2$',
                transform=ax_empirical_Esw.transAxes,
                c='white',
                bbox=dict(facecolor='black'),horizontalalignment='left')
    ax_empirical_Esw.text(0.12,0.94,r'$\circ R^2$',
                transform=ax_empirical_Esw.transAxes,
                c='black',horizontalalignment='left')

    ax_empirical_ampere.text(0.02,0.94,r'$\circ R^2$',
                transform=ax_empirical_ampere.transAxes,
                c='white',
                bbox=dict(facecolor='black'),horizontalalignment='left')
    ax_empirical_ampere.text(0.12,0.94,r'$\circ R^2$',
                transform=ax_empirical_ampere.transAxes,
                c='black',horizontalalignment='left')

    # Save
    figurename = f"{path}/figure5.png"
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)


#############################################################################

def main() -> None:
    ## Global variables and file paths
    global TINIT,TEND,TCUT,TMAIN,TMIN
    TINIT = dt.datetime(2024,5,10,13,0)
    TIMPACT = dt.datetime(2024,5,10,17)
    #TMAIN = dt.datetime(2024,5,10,17,54)
    TMAIN = dt.datetime(2024,5,10,17,30)
    TCUT = dt.datetime(2024,5,11,10,0)
    TMIN  = dt.datetime(2024,5,11,1,30)
    TEND  = dt.datetime(2024,5,11,17,0)
    inBase = os.path.realpath('..')+'/'
    inLogs = os.path.join(inBase,'data/logs/')
    inSats = os.path.join(inBase,'data/sat/')
    inAnalysis = os.path.join(inBase,'data/analysis/')
    outPath = os.path.join(inBase,'outputs/figures')
    unfiled = os.path.join(outPath,'unfiled')
    for path in [outPath,unfiled,]:
        os.makedirs(path,exist_ok=True)
    ## setting pyplot configurations
    plt.rcParams.update(pyplotsetup(mode='print'))

    ## Log Data
    dataset = {}
    dataset['obs'] = read_indices(inLogs,start=TINIT,
                                  end=TIMPACT+dt.timedelta(hours=24),
                                  read_supermag=False)
    dataset['obs']['pc'] =read_pc(f"{inBase}data/pc_index/pcnpcs4318007.txt")
    solarwind = dataset['obs']['swmf_sw']
    swmf_log = dataset['obs']['swmf_log']
    omni = dataset['obs']['omni']
    pc = dataset['obs']['pc']
    #magfile = "../data/large/GM/IO2/magnetometers_e20240510-130000.mag"
    #vmagnets = loadmagnetometers(magfile)
    magfile = "../data/logs/magnetometers_e20240510-130000.npz"
    mag = np.load(magfile,allow_pickle=True)
    vmagnets = pd.DataFrame(dict(mag))
    vmagnets.index = vmagnets['time']
    vmagnets.drop(columns='time',inplace=True)


    ## Analysis Data
    dataset['analysis'] = load_hdf_sort(inAnalysis+'energetics.h5')
    with pd.HDFStore(inAnalysis+'integrated_currents.h5') as store:
        dataset['analysis']['currents'] = store['/FAC']
    I_swmf = dataset['analysis']['currents']
    mp = dataset['analysis']['mpdict']['ms_full']
    closed = dataset['analysis']['msdict']['closed']
    lobes = dataset['analysis']['msdict']['lobes']
    plasmasheet = dataset['analysis']['msdict']['plasmasheet']
    inner = dataset['analysis']['inner_mp']
    ie = {}
    ie['N'] = dict(np.load("../data/large/IE/ionosphere/compiled_N.npz",
                           allow_pickle=True))
    ie['S'] = dict(np.load("../data/large/IE/ionosphere/compiled_S.npz",
                           allow_pickle=True))

    ## Satellite data
    sats = {}
    # GOES16
    goes_hardcopy = '../data/sat/goes_hardcopy.csv'
    if os.path.exists(goes_hardcopy):
        print(f'{goes_hardcopy} found ...')
        goes_df = pd.read_csv(goes_hardcopy,index_col='time')
        goes_df.index = pd.to_datetime(goes_df.index)
        goes_b = {'goes16':goes_df}
    else:
        goes_pos,goes_b,goes_plasma = collect_goes(mp.index[0],mp.index[-1],
                                               probes=['16'],writeData=False)
        goes_df = goes_b['goes16']
        goes_df.index.name = 'time'
        goes_df.to_csv(goes_hardcopy)
    for key,df in goes_b.items():
        sats[key] = df
    # THEMIS
    with pd.HDFStore(inSats+'themis_pos.h5') as store:
        sats['themisB'] = store['/themisB']
    ## Virtual satellite data
    vsatfiles = glob.glob(f'{inSats}*.sat')
    dataset['vsats'] = simdata_to_df(vsatfiles)
    themisA = dataset['vsats']['themisA']
    themisD = dataset['vsats']['themisD']
    themisE = dataset['vsats']['themisE']
    vsats = dataset['vsats']

    ## AMPERE data
    ampere_path = '../data/ampere'
    all_data = pd.DataFrame()
    for infile in glob.glob(f"{ampere_path}/*.dat"):
        df = read_currents(infile)
        all_data = pd.concat([all_data,df])
    all_data = all_data.replace(9999.00,np.nan)
    dataset['ampere'] = all_data
    I_ampere = dataset['ampere'].sort_index()

    ## DMSP
    dmsp = {}
    for sat in tqdm(['F16_N','F17_N','F18_N','F16_S','F17_S','F18_S']):
        dmsp[sat] = dict(np.load(f"../data/dmsp/compiled_{sat}.npz",
                           allow_pickle=True))

    ## SWIPE
    swipe = dict(np.load("../data/swipe/swipe_cpcp.npz"))

    ## Create Figures
    #plot_figure_1(unfiled,solarwind,swmf_log,mp,ie,omni)
    plot_figure_2(unfiled,mp,solarwind,sats,vsats,vmagnets,
                  I_ampere,I_swmf,pc,swmf_log,dmsp,ie,swipe)
    #plot_figure_3(unfiled,solarwind,mp,I_swmf,I_ampere,swmf_log)
    #plot_figure_4(unfiled,solarwind,mp,I_swmf,I_ampere,swmf_log)
    #plot_figure_5(unfiled,solarwind,mp,I_swmf,I_ampere,swmf_log,dmsp,ie,swipe)


if __name__ == "__main__":
    main()
