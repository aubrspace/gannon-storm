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
from matplotlib import pyplot as plt
from matplotlib import cm
from cmcrameri import cm as cm2
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

def K1label():
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
             label=r'n $\left[\#/cc\right]$',c='goldenrod')
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
                   mp:pd.DataFrame,**kwargs:dict) -> plt.Axes:
    K1     = (mp['K_netK1 [W]']+mp['UtotM1 [W]'])/-1e12
    K1_ave = (mp['K_netK1 [W]']+mp['UtotM1 [W]']).rolling('600s').mean()/-1e12
    ax.plot(solarwind.index,solarwind['Esw']/1e3,label=r'$E_{KL}$',
            c='black',lw=4)
    ax.fill_between(solarwind.index,solarwind['EinWang']/1e12,
                    label='$E_{in}$',fc='grey')
    #ax.plot(mp.index,K1,c='plum',label='_K1raw',alpha=0.8)
    ax.plot(mp.index,K1_ave,c='magenta',label=K1label())
    return ax

def draw_dst_panel(ax:plt.Axes,swmf_log:pd.DataFrame,
                   omni:pd.DataFrame,**kwargs:dict) -> plt.Axes:
    ax.plot(omni.index,omni['sym_h'],label='OMNI',c='black',lw=3)
    ax.plot(swmf_log.index,swmf_log['dst_sm'],label='SWMF',c='magenta',lw=1.5)
    return ax

def plot_figure_1(path:str,solarwind:pd.DataFrame,
                            swmf_log:pd.DataFrame,
                                  mp:pd.DataFrame,
                                omni:pd.DataFrame,**kwargs:dict) -> plt.Axes:
    # Figure
    fig,axes = plt.subplots(4,figsize=[24,32],sharex=True)
    # Plots
    axes[0] = draw_imf_panel(axes[0],solarwind)
    axes[1] = draw_plasma_panel(axes[1],solarwind,mp)
    axes[2] = draw_Esw_panel(axes[2],solarwind,mp)
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
    return ax

def draw_magnetometer_panel(ax:plt.Axes,vmagnets:pd.DataFrame,
                                                  station:str) -> plt.Axes:
    # Call supermag to get the obs data
    #   NOTE pass by reference here, supermag is a global dict
    supermag = proc_supermag(station,vmagnets.index[0],vmagnets.index[-1])
    swmf_single = vmagnets[vmagnets['IAGA']==station]
    # Draw both lines on this axis with some settings
    ax.plot(supermag.index,supermag['dBn'],label='dBn_sm',c='black',lw=3)
    plot_colorline(swmf_single.index,swmf_single['dBn'].values,
                   swmf_single['mlt'].values,ax)
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
                   I_swmf:pd.DataFrame,**kwargs:dict) -> plt.Axes:
    # Get correlation coefficient for model/data
    t_swmf = [float(t.to_numpy()) for t in I_swmf.index-TMIN]
    t_ampere = [float(t.to_numpy()) for t in I_ampere.index-TMIN]

    x_north = np.interp(t_swmf,t_ampere,
                        I_ampere['I_total_up_South_[MA]'].values)
    x_south = np.interp(t_swmf,t_ampere,
                        I_ampere['I_total_up_South_[MA]'].values)

    X_north = np.column_stack((x_north,np.ones(len(x_north))))
    X_south = np.column_stack((x_south,np.ones(len(x_south))))

    y_north = I_swmf['up_north_MA'].values
    y_south = I_swmf['up_south_MA'].values

    model_north = sm.OLS(y_north,X_north)
    model_south = sm.OLS(y_south,X_south)

    result_north = model_north.fit()
    result_south = model_south.fit()

    # Plot
    ax.plot(I_swmf.index,I_swmf['up_north_MA'],label='SWMF_N',
             c='blue')
    ax.plot(I_swmf.index,I_swmf['up_south_MA'],label='SWMF_S',
             c='purple')

    ax.axhline(0,c='grey',lw=1)

    ax.plot(I_ampere.index,I_ampere['I_total_up_North_[MA]'],
             label='AMPERE_N',c='red')
    ax.plot(I_ampere.index,I_ampere['I_total_up_South_[MA]'],
             label='AMPERE_S',c='orange')
    ax.text(0.99,0.92,r'$R^2$'+f'={result_north.rsquared:.2f}',
             transform=ax.transAxes,c='blue',horizontalalignment='right')
    ax.text(0.99,0.84,r'$R^2$'+f'={result_south.rsquared:.2f}',
             transform=ax.transAxes,c='purple',horizontalalignment='right')
    return ax

def draw_CPCP_panel(ax:plt.Axes,pc:pd.DataFrame,
                    swmf_log:pd.DataFrame,**kwargs:dict) -> plt.Axes:
    ax.plot(pc.index,pc['cpcpn'],label='Ridley&Kihn',c='black',lw=3)
    ax.plot(pc.index,pc['cpcps'],label='_Ridley&Kihn',c='black',lw=1.5,
                                                                      ls='--')
    ax.plot(swmf_log.index,swmf_log['cpcpn'],label='SWMF_N',c='magenta',lw=3)
    ax.plot(swmf_log.index,swmf_log['cpcps'],label='_SWMF_S',c='magenta',
            lw=1.5,ls='--')
    return ax

def plot_figure_2(path:str,sats:pd.DataFrame,
                          vsats:pd.DataFrame,
                       vmagnets:pd.DataFrame,
                       I_ampere:pd.DataFrame,
                         I_swmf:pd.DataFrame,
                             pc:pd.DataFrame,
                       swmf_log:pd.DataFrame,**kwargs:dict) -> plt.Axes:
    stations = ['FMC','MEA','T43']
    ampere_path = '../data/ampere/'
    ampere_quicklook='1715372880.north.png'
    ampere_image = plt.imread(f"{ampere_path}{ampere_quicklook}")
    paraview_path = '../outputs/vis/'
    paraview_compare = 'FAC_ampere_compare.png'
    paraview_image = plt.imread(f"{paraview_path}{paraview_compare}")
    # Figure
    fig = plt.figure(figsize=[24,32])
    # GridSpecs #TODO reduce whitespace
    fivepiece = plt.GridSpec(4,1,hspace=0.1,figure=fig,
                             left=0.09,right=0.95,bottom=0.04,top=0.98,
                             height_ratios=[1,2,1,1])
    middleSection = fivepiece[1].subgridspec(1,2,hspace=0.1,wspace=0.01,
                                             width_ratios=[3,1])
    magPanels = middleSection[0].subgridspec(3,1,hspace=0.05)
    polarPanels = middleSection[1].subgridspec(2,1,hspace=0.01)
    # Declare axes
    sat_ax = fig.add_subplot(fivepiece[0])
    mag_ax1 = fig.add_subplot(magPanels[0])
    mag_ax2 = fig.add_subplot(magPanels[1])
    mag_ax3 = fig.add_subplot(magPanels[2])
    pole_ax1 = fig.add_subplot(polarPanels[0])
    pole_ax2 = fig.add_subplot(polarPanels[1])
    fac_ax = fig.add_subplot(fivepiece[2])
    cpcp_ax = fig.add_subplot(fivepiece[3])
    # Plot
    sat_ax  = draw_vsat_panel(sat_ax,sats,vsats)
    mag_ax1 = draw_magnetometer_panel(mag_ax1,vmagnets,stations[0])
    mag_ax2 = draw_magnetometer_panel(mag_ax2,vmagnets,stations[1])
    mag_ax3 = draw_magnetometer_panel(mag_ax3,vmagnets,stations[2])
    pole_ax1.imshow(ampere_image)
    pole_ax2.imshow(paraview_image)
    fac_ax  = draw_FAC_panel(fac_ax,I_ampere,I_swmf)
    cpcp_ax = draw_CPCP_panel(cpcp_ax,pc,swmf_log)
    # Decorate
    general_plot_settings(sat_ax,do_xlabel=False,legend=True,
                          legend_loc='lower left',
                          ylabel=r'$B_Z\left[nT\right]$',
                          xlim=[TINIT,TEND],timedelta=False)
    sat_ax.text(0.99,0.2,f"(a)",transform=sat_ax.transAxes,
                  c='black',horizontalalignment='right',fontsize=36)
    letters = ['(b)','(c)','(d)']
    for i,ax in enumerate([mag_ax1,mag_ax2,mag_ax3]):
        general_plot_settings(ax,do_xlabel=False,legend=False,
                              ylabel=f'{stations[i]}',
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
    pole_ax1.text(0.98,0.02,f"(e)",transform=pole_ax1.transAxes,
                  c='black',horizontalalignment='right',fontsize=36)
    pole_ax2.axis('off')
    pole_ax2.text(0.98,0.02,f"(f)",transform=pole_ax2.transAxes,
                  c='black',horizontalalignment='right',fontsize=36)
    general_plot_settings(fac_ax,do_xlabel=False,legend=True,
                          legend_loc='upper left',
                          ylabel=r'$\int$FAC $\left[MA\right]$',
                          xlim=[TINIT,TEND], timdelta=False)
    fac_ax.set_xticklabels([])
    fac_ax.text(0.98,0.02,f"(g)",transform=fac_ax.transAxes,
                  c='black',horizontalalignment='right',fontsize=36)
    general_plot_settings(cpcp_ax,do_xlabel=True,legend=True,
                          xlim=[TINIT,TEND],ylim=[0,700],
                          ylabel=r'CPCP $\left[kV\right]$',timedelta=False)
    cpcp_ax.text(0.98,0.02,f"(h)",transform=cpcp_ax.transAxes,
                  c='black',horizontalalignment='right',fontsize=36)
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
               scat_color:np.ndarray, **kwargs:dict) ->[plt.Axes,plt.scatter]:
    # Get Pearson r
    slope,intercept,r,p,std_err = scipy.stats.linregress(X.values,Y.values)
    # Obtain low,50, and high %tiles, and variance binned by our X axis
    X_bins   = np.linspace(X.quantile(0.005),X.quantile(0.995),33)
    bin_Ranges = bin_and_describe(X,Y,Y,X_bins,0.05,0.95)
    # Plot
    if kwargs.get('text_loc','left')=='left':
        ax.text(0.02,0.94,r'$R^2$'+f'={r**2:.2f}',transform=ax.transAxes,
                                      c='dimgrey',horizontalalignment='left')
    elif kwargs.get('text_loc','left')=='right':
        ax.text(0.98,0.94,r'$R^2$'+f'={r**2:.2f}',transform=ax.transAxes,
                                     c='dimgrey',horizontalalignment='right')
    extended_fill_between(ax,X_bins,bin_Ranges['pLow_all'],
                                    bin_Ranges['pHigh_all'],'gold',0.2)
    sc = ax.scatter(X,Y,cmap=cm.winter,c=scat_color,
                        s=50,alpha=0.8)
    ax.plot(X_bins,bin_Ranges['p50_all'],c='black',lw=4)
    ax.plot(X_bins,slope*X_bins+intercept,c='grey',ls='--',lw=3)
    return ax,sc

def draw_fit_resid_panel(ax:plt.Axes,
                            x:np.ndarray,
                            y:np.ndarray,
                   scat_color:np.ndarray,
                           **kwargs:dict)->[plt.Axes,plt.scatter]:
    x_sort = x.argsort()
    x = x[x_sort]
    # Create a model for prediction using Ordinary Least Squares regression
    if kwargs.get('form','linear')=='linear':
        X = np.column_stack((x,np.ones(len(x))))
    elif kwargs.get('form','linear')=='sq':
        X = np.column_stack((x,x**(1/2),np.ones(len(x))))
    elif kwargs.get('form','linear')=='cu':
        X = np.column_stack((x,x**(1/3),np.ones(len(x))))
    y = y[x_sort]

    model  = sm.OLS(y,X)
    result = model.fit()

    sc = ax.scatter(result.fittedvalues,
                    (result.fittedvalues-y)/y,
                    s=50,alpha=0.8,cmap=cm.managua,c=scat_color[x_sort])
    ax.axhline(0,c='grey')
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

'''
def plot_figure_3_old(path:str,solarwind:pd.DataFrame,
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
    X3      = np.column_stack((x2,x2**(1/2),np.ones(len(x2))))
    y3      = CPCP.iloc[x3_sort]

    model3  = sm.OLS(y3,X3)
    result3 = model3.fit()


    # Figure
    fig = plt.figure(figsize=[32,30])
    # GridSpecs
    sliver = plt.GridSpec(1,2,hspace=0.1,figure=fig,
                             left=0.06,right=0.92,bottom=0.04,top=0.98,
                             width_ratios=[56,1],wspace=0.1)
    mainsplit = sliver[0].subgridspec(1,2,width_ratios=[1.7,6.3],wspace=0.15)
    fourstack = mainsplit[0].subgridspec(4,1,hspace=0.22)
    sixpack = mainsplit[1].subgridspec(3,2,hspace=0.17,wspace=0.1)

    # Declare axes
    fit_ax1 = fig.add_subplot(fourstack[0])
    fit_ax2 = fig.add_subplot(fourstack[1])
    fit_ax3 = fig.add_subplot(fourstack[2])
    fit_ax4 = fig.add_subplot(fourstack[3])

    model_ax1 = fig.add_subplot(sixpack[0,0])
    model_ax2 = fig.add_subplot(sixpack[0,1])
    model_ax3 = fig.add_subplot(sixpack[1,0])
    model_ax4 = fig.add_subplot(sixpack[1,1])
    model_ax5 = fig.add_subplot(sixpack[2,0])
    model_ax6 = fig.add_subplot(sixpack[2,1])

    # Plot
    fit_ax1,sc = draw_scatter_panel(fit_ax1,Esw,K1,pdyn,text_loc='right')
    fit_ax2,sc = draw_scatter_panel(fit_ax2,K1[K1>0],FAC[K1>0],pdyn[K1>0])
    fit_ax3,sc = draw_scatter_panel(fit_ax3,FAC,CPCP,pdyn)
    fit_ax4,sc = draw_scatter_panel(fit_ax4,Ein,CPCP,pdyn)

    model_ax1,sc = draw_scatter_panel(model_ax1,BOYLE,CPCP,pdyn,
                                      text_loc='right')
    model_ax3,sc = draw_scatter_panel(model_ax3,SHILL,CPCP,pdyn)
    model_ax5,sc = draw_scatter_panel(model_ax5,KRID,CPCP,pdyn,
                                      text_loc='right')

    model_ax2,sc = draw_scatter_panel(model_ax2,Esw,CPCP,pdyn,
                                      text_loc='right')
    model_ax4,sc = draw_scatter_panel(model_ax4,K1[K1>0]**0.5,CPCP[K1>0],
                                      pdyn[K1>0])
    model_ax6,sc = draw_scatter_panel(model_ax6,AMP_FAC,CPCP,pdyn)
    # Decorate
    cb_ax = fig.add_axes([0.92,.084,.02,.854])
    #cb_ax = fig.add_axes([0.88,.124,.04,.754])
    cb    = fig.colorbar(sc,orientation='vertical',cax=cb_ax)
    cb.set_label(r"$p_{dyn}\left[nPa\right]$",fontsize=36)
    fit_ax1.set_xlabel(r"$E_{in}\left[TW\right]$ Wang'14")
    fit_ax1.set_ylabel(K1label())

    fit_ax2.set_xlabel(K1label())
    fit_ax2.set_ylabel(r'$\int$FAC $\left[MA\right]$')
    fit_ax2.plot(x,result_sq.fittedvalues,c='red',ls='--')
    fit_ax2.text(0.02,0.88,r'$R^2$'+f'={result_sq.rsquared:.2f}',
             transform=fit_ax2.transAxes,c='red',horizontalalignment='left')

    fit_ax3.set_xlabel(r'$\int$FAC $\left[MA\right]$')
    fit_ax3.set_ylabel(r'CPCP $\left[kV\right]$')

    fit_ax4.set_xlabel(r"$E_{in}\left[TW\right]$ Wang'14")
    fit_ax4.set_ylabel(r'CPCP $\left[kV\right]$')
    fit_ax4.plot(x2,result2_sq.fittedvalues,c='red',ls='--')
    fit_ax4.text(0.02,0.88,r'$R^2$'+f'={result2_sq.rsquared:.2f}',
             transform=fit_ax4.transAxes,c='red',horizontalalignment='left')

    model_ax1.set_xlabel(r"Boyle'97 CPCP $\left[kV\right]$")
    model_ax3.set_xlabel(r"Siscoe-Hill'02 CPCP $\left[kV\right]$")
    model_ax5.set_xlabel(r"Kivelson-Ridley'08 CPCP $\left[kV\right]$")

    model_ax2.set_xlabel(r'$E_{KL}\left[mA/m\right]$')
    model_ax2.plot(x3,result3.fittedvalues,c='red',ls='--')
    model_ax2.text(0.98,0.88,r'$R^2$'+f'={result3.rsquared:.2f}',
             transform=model_ax2.transAxes,c='red',horizontalalignment='right')
    model_ax4.set_xlabel(
                    r"$\sqrt{\overline{\int_O{\mathbf{K}\cdot\mathbf{n}}}}$")
    model_ax6.set_xlabel(r'AMPERE $\int$FAC $\left[MA\right]$')


    for i,ax in enumerate([model_ax1,model_ax2,model_ax3,
                           model_ax4,model_ax5,model_ax6]):
        ax.margins(x=0.01,y=0.01)
        ax.set_ylabel(r'CPCP $\left[kV\right]$')
        if i==1 or i==3 or i==5:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
    fig.patches.extend([plt.Rectangle([0,0.25],0.23,1,fill=True,fc='plum',
                   alpha=0.2,zorder=-1,transform=fig.transFigure,figure=fig)])
    fig.patches.extend([plt.Rectangle([0,0],0.23,0.25,fill=True,fc='grey',
                   alpha=0.5,zorder=-1,transform=fig.transFigure,figure=fig)])

    # Save
    figurename = f"{path}/figure3.png"
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)
'''

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
    #from IPython import embed; embed()

    # Create some models for how the K1 FAC relationship could be fit
    x = K1[K1>0].values
    x_sort = x.argsort()
    x = x[x_sort]
    X_sq = np.column_stack((x,x**(1/2),np.ones(len(x))))
    y = FAC[K1>0].iloc[x_sort]

    model_sq = sm.OLS(y,X_sq)
    result_sq     = model_sq.fit()

    # For Ein predict CPCP
    x2      = Ein.values
    x2_sort = x2.argsort()
    x2      = x2[x2_sort]
    X2_sq   = np.column_stack((x2,x2**(1/2),np.ones(len(x2))))
    y2      = CPCP.iloc[x2_sort]

    model2_sq  = sm.OLS(y2,X2_sq)
    result2_sq = model2_sq.fit()

    # Figure
    fig = plt.figure(figsize=[28,30])
    # GridSpecs #TODO reduce whitespace
    slivers = plt.GridSpec(1,3,hspace=0.1,figure=fig,
                             left=0.06,right=0.92,bottom=0.04,top=0.98,
                             width_ratios=[4,56,4],wspace=0.1)
    eightpack = slivers[1].subgridspec(4,2,hspace=0.2,wspace=0.1)
    # Declare axes
    ax1 = fig.add_subplot(eightpack[0])
    ax2 = fig.add_subplot(eightpack[1])
    ax3 = fig.add_subplot(eightpack[2])
    ax4 = fig.add_subplot(eightpack[3])
    ax5 = fig.add_subplot(eightpack[4])
    ax6 = fig.add_subplot(eightpack[5])
    ax7 = fig.add_subplot(eightpack[6])
    ax8 = fig.add_subplot(eightpack[7])

    # Plot
    ax1,scl = draw_scatter_panel(ax1,Esw,K1,pdyn,text_loc='right')
    ax3,scl = draw_scatter_panel(ax3,K1[K1>0],FAC[K1>0],pdyn[K1>0])
    ax5,scl = draw_scatter_panel(ax5,FAC,CPCP,pdyn)
    ax7,scl = draw_scatter_panel(ax7,Ein,CPCP,pdyn)

    ax2,scr = draw_fit_resid_panel(ax2,Ein.values,K1.values,times)
    ax4,scr = draw_fit_resid_panel(ax4,K1[K1>0],FAC[K1>0],
                                        times[K1>0],form='sq')
    ax6,scr = draw_fit_resid_panel(ax6,FAC.values,CPCP.values,times)
    ax8,scr = draw_fit_resid_panel(ax8,Ein.values,CPCP.values,times,form='sq')

    # Decorate
    cb_axl = fig.add_axes([0.06,.084,.02,.854])
    cb_axr = fig.add_axes([0.92,.084,.02,.854])
    cbl = fig.colorbar(scl,orientation='vertical',cax=cb_axl)
    cbr = fig.colorbar(scr,orientation='vertical',cax=cb_axr)
    cbl.set_label(r"$p_{dyn}\left[nPa\right]$",fontsize=36)
    cbr.set_label(r"$T-$ 11-01:30:00 $\left[Hr\right]$",fontsize=36)

    cb_axl.yaxis.set_ticks_position("left")
    cb_axl.yaxis.set_label_position("left")

    ax1.set_xlabel(r"$E_{in}\left[TW\right]$ Wang'14")
    ax1.set_ylabel(K1label())

    ax3.set_xlabel(K1label())
    ax3.set_ylabel(r'$\int$FAC $\left[MA\right]$')
    ax3.plot(x,result_sq.fittedvalues,c='red',ls='--')
    ax3.text(0.02,0.88,r'$R^2$'+f'={result_sq.rsquared:.2f}',
             transform=ax3.transAxes,c='red',horizontalalignment='left')

    ax5.set_xlabel(r'$\int$FAC $\left[MA\right]$')
    ax5.set_ylabel(r'CPCP $\left[kV\right]$')

    ax7.set_xlabel(r"$E_{in}\left[TW\right]$ Wang'14")
    ax7.set_ylabel(r'CPCP $\left[kV\right]$')
    ax7.plot(x2,result2_sq.fittedvalues,c='red',ls='--')
    ax7.text(0.02,0.88,r'$R^2$'+f'={result2_sq.rsquared:.2f}',
             transform=ax7.transAxes,c='red',horizontalalignment='left')

    ax2.set_xlabel(r'predict. '+K1label())
    ax2.set_ylabel(r'$E_{in}$ Fit Resid.')
    ax2.set_ylim([-100,50])

    ax4.set_xlabel(r'predict. $\int$FAC $\left[MA\right]$')
    ax4.set_ylabel(K1label()+' (sqrt) Fit Resid.')

    ax6.set_xlabel(r'predict. CPCP $\left[kV\right]$')
    ax6.set_ylabel(r'$FAC$ Fit Resid.')

    ax8.set_xlabel(r'predict. CPCP $\left[kV\right]$')
    ax8.set_ylabel(r'$E_{in}$ (sqrt) Fit Resid.')

    letters = ['(a)','(e)','(b)','(f)','(c)','(g)','(d)','(h)']
    for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]):
        if i==1 or i==3 or i==5 or i==7:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            if i!=1:
                ax.set_ylim([-1,8])
            else:
                ax.axhline(-1,c='grey',ls='--')
                ax.axhline(8,c='grey',ls='--')
        ax.text(0.98,0.02,f"{letters[i]}",transform=ax.transAxes,
                c='black',horizontalalignment='right',fontsize=36)
    #fig.patches.extend([plt.Rectangle([0,0.25],0.23,1,fill=True,fc='plum',
    #               alpha=0.2,zorder=-1,transform=fig.transFigure,figure=fig)])

    # Save
    figurename = f"{path}/figure3.png"
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)

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


'''
def plot_figure_4_old(path:str,solarwind:pd.DataFrame,
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
    K1  = ((mp['K_netK1 [W]']+mp['UtotM1 [W]']).rolling('600s'
                                                       ).mean()/-1e12).values
    Ein     = np.interp(t_mp,t_sw,solarwind['EinWang'].values/1e12)
    Esw     = np.interp(t_mp,t_sw,solarwind['Esw'].values/1e3)
    CPCP    = np.interp(t_mp,t_log,swmf_log['cpcpn'].values)
    FAC     = np.interp(t_mp,t_ie,(I_swmf['up_north_MA']).values)
    AMP_FAC = np.interp(t_mp,t_ampere,I_ampere['I_total_up_North_[MA]'].values)
    pdyn    = np.interp(t_mp,t_sw,solarwind['pdyn'].values)
    times   = np.array([t/3600e9 for t in t_mp])
    BOYLE   = np.interp(t_mp,t_sw,solarwind['CPCP_B97'].values)
    SHILL   = np.interp(t_mp,t_sw,solarwind['CPCP_S02'].values)
    KRID    = np.interp(t_mp,t_sw,solarwind['CPCP_K08'].values)


    # Figure
    fig = plt.figure(figsize=[32,30])
    # GridSpecs #TODO reduce whitespace
    sliver = plt.GridSpec(1,2,hspace=0.1,figure=fig,
                             left=0.06,right=0.92,bottom=0.04,top=0.98,
                             width_ratios=[56,1],wspace=0.1)
    mainsplit = sliver[0].subgridspec(1,2,width_ratios=[1.7,6.3],wspace=0.15)
    fourstack = mainsplit[0].subgridspec(4,1,hspace=0.2)
    sixpack = mainsplit[1].subgridspec(3,2,hspace=0.05,wspace=0.1)
    # Declare axes
    fit_ax1 = fig.add_subplot(fourstack[0])
    fit_ax2 = fig.add_subplot(fourstack[1])
    fit_ax3 = fig.add_subplot(fourstack[2])
    fit_ax4 = fig.add_subplot(fourstack[3])

    model_ax1 = fig.add_subplot(sixpack[0,0])
    model_ax2 = fig.add_subplot(sixpack[0,1])
    model_ax3 = fig.add_subplot(sixpack[1,0])
    model_ax4 = fig.add_subplot(sixpack[1,1])
    model_ax5 = fig.add_subplot(sixpack[2,0])
    model_ax6 = fig.add_subplot(sixpack[2,1])

    # Plot
    fit_ax1,sc = draw_fit_resid_panel(fit_ax1,Ein,K1,times)
    fit_ax2,sc = draw_fit_resid_panel(fit_ax2,K1[K1>0],FAC[K1>0],
                                        times[K1>0],form='sq')
    fit_ax3,sc = draw_fit_resid_panel(fit_ax3,FAC,CPCP,times)
    fit_ax4,sc = draw_fit_resid_panel(fit_ax4,Ein,CPCP,times,form='sq')

    model_ax1,sc = draw_model_resid_panel(model_ax1,BOYLE,CPCP,times)
    model_ax3,sc = draw_model_resid_panel(model_ax3,SHILL,CPCP,times)
    model_ax5,sc = draw_model_resid_panel(model_ax5,KRID,CPCP,times)

    model_ax2,sc = draw_fit_resid_panel(model_ax2,Esw,CPCP,times)
    model_ax4,sc = draw_fit_resid_panel(model_ax4,K1[K1>0],CPCP[K1>0],
                                        times[K1>0],form='sq')
    model_ax6,sc = draw_fit_resid_panel(model_ax6,AMP_FAC,CPCP,times)
    # Decorate
    cb_ax = fig.add_axes([0.92,.084,.02,.854])
    #cb_ax = fig.add_axes([0.88,.124,.04,.754])
    cb = fig.colorbar(sc,orientation='vertical',cax=cb_ax)
    #cb.set_label(r"$p_{dyn}\left[nPa\right]$",fontsize=36)
    cb.set_label(r"$T-$ 11-01:30:00 $\left[Hr\right]$",fontsize=36)

    fit_ax1.set_xlabel(r'predict. '+K1label())
    fit_ax1.set_ylabel(r'$E_{in}$ Fit Resid.')
    fit_ax1.set_ylim([-100,50])

    fit_ax2.set_xlabel(r'predict. $\int$FAC $\left[MA\right]$')
    fit_ax2.set_ylabel(K1label()+' (sqrt) Fit Resid.')

    fit_ax3.set_xlabel(r'predict. CPCP $\left[kV\right]$')
    fit_ax3.set_ylabel(r'$FAC$ Fit Resid.')

    fit_ax4.set_xlabel(r'predict. CPCP $\left[kV\right]$')
    fit_ax4.set_ylabel(r'$E_{in}$ (sqrt) Fit Resid.')

    model_ax1.set_ylabel(r"Boyle'97 - SWMF")
    model_ax3.set_ylabel(r"Siscoe-Hill'02 - SWMF")
    model_ax5.set_ylabel(r"Kivelson-Ridley'08 - SWMF")
    model_ax5.set_xlabel(r"Predicted CPCP $\left[kV\right]$")

    model_ax2.set_ylabel(r"$E_{KL}$ - SWMF")
    model_ax4.set_ylabel(
            r"$\sqrt{\overline{\int_O{\mathbf{K}\cdot\mathbf{n}}}}$ - SWMF")
    model_ax6.set_ylabel(r"AMPERE $FAC$ - SWMF")
    model_ax6.set_xlabel(r"Predicted CPCP $\left[kV\right]$")

    for i,ax in enumerate([model_ax1,model_ax2,model_ax3,
                           model_ax4,model_ax5,model_ax6]):
        if i==1 or i==3 or i==5:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        else:
            ax.margins(x=0.01)
        if i!=4 and i!=5:
            ax.set_xticklabels([])
        ax.set_xlim([0,800])
        ax.set_ylim([-2,6])
    fig.patches.extend([plt.Rectangle([0,0.25],0.23,1,fill=True,fc='plum',
                   alpha=0.2,zorder=-1,transform=fig.transFigure,figure=fig)])
    fig.patches.extend([plt.Rectangle([0,0],0.23,0.25,fill=True,fc='grey',
                   alpha=0.5,zorder=-1,transform=fig.transFigure,figure=fig)])

    # Save
    figurename = f"{path}/figure4.png"
    fig.savefig(figurename)
    plt.close(fig)
    print('\033[92m Created\033[00m',figurename)

'''
#############################################################################

def main() -> None:
    ## Global variables and file paths
    global TINIT,TEND,TCUT,TMIN
    TINIT = dt.datetime(2024,5,10,13,0)
    TIMPACT = dt.datetime(2024,5,10,17)
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
    I_ampere = dataset['ampere']

    ## Create Figures
    #plot_figure_1(unfiled,solarwind,swmf_log,mp,omni)
    plot_figure_2(unfiled,sats,vsats,vmagnets,I_ampere,I_swmf,pc,swmf_log)
    #plot_figure_3(unfiled,solarwind,mp,I_swmf,I_ampere,swmf_log)
    #plot_figure_4(unfiled,solarwind,mp,I_swmf,I_ampere,swmf_log)


if __name__ == "__main__":
    main()
