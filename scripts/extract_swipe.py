#!/usr/bin/env python3
""" Extract ionosphere solution from empirical SWIPE model for this event
"""
import os,sys,glob,time
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
#
from global_energetics.analysis.proc_indices import read_indices
from pyswipe import SWIPE

def get_time(infile:str) -> dt.datetime:
    format_string = "3d__paraview_1_e%Y%m%d-%H%M%S-%f.aux"
    return dt.datetime.strptime(infile.split('/')[-1],format_string)

def add_Btilt_to_solarwind(solarwind:pd.DataFrame,
                             aux_dir:str) -> pd.DataFrame:
    # Gather and sort files by their timestamps in the file names
    filelist = sorted(glob.glob(aux_dir+"*.aux"),key=get_time)
    # Initialize arrays
    t_btilt = np.zeros(len(filelist))
    btilt   = np.zeros(len(filelist))
    for i,infile in enumerate(filelist):
        with open(infile,'r') as f:
            raw = f.readlines()
        # Read Btilt from the aux files
        aux = {}
        [aux.update({l.split(':')[0]:l.split(':')[-1]}) for l in raw]
        # Infer the time from the filename
        intime = dt.datetime.strptime(infile.split('/')[-1],
                                       "3d__paraview_1_e%Y%m%d-%H%M%S-%f.aux")
        t_btilt[i]  = (intime-TINIT).total_seconds()
        btilt[i] = float(aux['BTHETATILT'].split()[0])
    # Interpolate flux into our solarwind timeseries
    t_sw = [(t-TINIT).total_seconds() for t in solarwind.index]
    solarwind['btilt'] = np.interp(t_sw,t_btilt,btilt)
    return solarwind

def add_f107_to_solarwind(solarwind:pd.DataFrame,
                          f107_file:str) -> pd.DataFrame:
    # Read f10.7
    with open(f107_file,'r') as f:
        raw = f.readlines()
    split_data = [l.split() for l in raw[raw.index("#START\n")+1::]]
    times = np.array([dt.datetime(int(l[0]),int(l[1]),int(l[2]),
                                  int(l[3]),int(l[4])) for l in split_data])
    flux = np.array([float(l[-1]) for l in split_data])
    # Interpolate flux into our solarwind timeseries
    t_107 = [(t-TINIT).total_seconds() for t in times]
    t_sw = [(t-TINIT).total_seconds() for t in solarwind.index]
    solarwind['f107'] = np.interp(t_sw,t_107,flux)
    return solarwind

def run_swipe(solarwind:pd.DataFrame,path:str) -> None:
    # Set inputs
    velocity = solarwind['v'].values
    By = solarwind['by'].values
    Bz = solarwind['bz'].values
    Btilt = solarwind['btilt'].values * -1
    f107 = solarwind['f107'].values
    cpcp_n = np.zeros(len(f107))
    cpcp_s = np.zeros(len(f107))
    times = np.array([dt.datetime.strptime(str(t),"%Y-%m-%d %H:%M:%S")
                                                    for t in solarwind.index])
    print("Running SWIPE ...")
    for im in tqdm(range(0,len(velocity))):
        model = SWIPE(velocity[im],By[im],Bz[im],Btilt[im],f107[im])
        '''
        grid = model.scalargrid
        #NOTE using hard numbers here
        mlat_north = grid[0][0:10000].reshape(100,100)
        mlat_south = grid[0][10000::].reshape(100,100)
        mlat_north = grid[1][0:10000].reshape(100,100)
        mlat_south = grid[1][10000::].reshape(100,100)
        '''
        pot = model.get_potential()
        #pot_north = pot[0:10000].reshape(100,100)
        #pot_south = pot[0:10000].reshape(100,100)
        
        # Calc cpcp 
        cpcp_n[im] = pot[0:10000].max()-pot[0:10000].min()
        cpcp_s[im] = pot[10000::].max()-pot[10000::].min()
        # Make a plot
        #TODO
    result = {'time':times,'cpcp_n':cpcp_n,'cpcp_s':cpcp_s}
    np.savez_compressed(f"{path}/swipe_cpcp.npz",**result)
    print(f'\033[92m Created\033[00m {path}/swipe_cpcp.npz')

def main() -> None:
    inLogs = "../data/logs/"
    # Read in log data
    logs = read_indices(inLogs,start=TINIT,
                                 end=TIMPACT+dt.timedelta(hours=24),
                                  read_supermag=False)
    solarwind = logs['swmf_sw']
    solarwind = add_f107_to_solarwind(solarwind,"../data/logs/f107.txt")
    solarwind = add_Btilt_to_solarwind(solarwind,"../data/large/GM/IO2/")

    run_swipe(solarwind,"../data/swipe")
    
    return

if __name__ == "__main__":
    start_time = time.time()
    global TINIT,TEND,TCUT,TIMPACT,TMIN
    TINIT = dt.datetime(2024,5,10,13,0)
    TEND  = dt.datetime(2024,5,11,17,0)
    TCUT = dt.datetime(2024,5,11,10,0)
    TIMPACT = dt.datetime(2024,5,10,17)
    TMIN  = dt.datetime(2024,5,11,1,30)

    main()
    #timestamp
    ltime = time.time()-start_time
    print('--- {:d}min {:.2f}s ---'.format(int(ltime/60),
                                           np.mod(ltime,60)))
