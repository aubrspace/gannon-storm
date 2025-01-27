#!/usr/bin/env python3
"""
"""
#General file IO/debugging
import os,sys,glob,time
sys.path.append(os.getcwd().split('swmf-energetics')[0]+
                                      'swmf-energetics/')
#The standards for math and data handling
import datetime as dt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#NASA tools
from cdasws import CdasWs
cdas = CdasWs()
from sscws.sscws import SscWs
ssc = SscWs()
from sscws.coordinates import CoordinateSystem as coordsys
#Geopack for coord transforms
from geopack import geopack as gp
#SWMFpy
from swmfpy.io import write_imf_input
#Custom packages for calling CDAweb/OMNI + post processing
from global_energetics.wind_to_swmfInput import (collect_themis,collect_mms)

def clean_data(probedata):
    cleaned_dict = {'loc':pd.DataFrame(),
                    'imf':pd.DataFrame(),
                    'plasma':pd.DataFrame()}
    # Quality correct the data
    plasma = probedata['plasma']
    ######Plasma
    ######### N
    diff = abs(plasma['ne'].diff())
    diff.fillna(0)
    maxdiff = diff.max()
    i=0
    while maxdiff>45:
        plasma = plasma[diff<45]
        diff = abs(plasma['ne'].diff())
        diff.fillna(0)
        maxdiff = diff.max()
        i+=1
        if i>3000:
            maxdiff = 0
    print(f'N clean iter:{i}')
    N = plasma['ne']
    #N = plasma['n']
    P = plasma['p']
    Vx = plasma['vx']
    Vy = plasma['vy']
    Vz = plasma['vz']
    V = np.sqrt(Vx**2+Vy**2+Vz**2)
    Bx = probedata['imf']['bx']
    By = probedata['imf']['by']
    Bz = probedata['imf']['bz']
    # avoid slow and positive vx
    tdelta = [t.total_seconds() for t in V.index[1::]-V.index[0:-1]]
    dVdt = V.diff()[1::]/tdelta
    #condition1 = (shifted_data['vx']<-300)&(shifted_data['vx']>-1000)
    vcondition = ((Vx<-300) & (abs(Vy)<200) & (abs(Vz)<200) &
                  (Vx**2>(Vy**2+Vz**2)) & (abs(V.diff())<20))
    vcondition = ((vcondition) & (plasma.index<dt.datetime(2024,5,13,9,0)) |
                                 (plasma.index>dt.datetime(2024,5,13,9,15)))
    # avoid negative pressure and density
    tcondition = (P>0) & (N>1)
    conditions = vcondition & tcondition
    plasma = plasma[conditions]
    ######### Vx
    diff = abs(plasma['vx'].diff())
    diff.fillna(0)
    maxdiff = diff.max()
    i=0
    while maxdiff>200:
        plasma = plasma[diff<200]
        diff = abs(plasma['vx'].diff())
        diff.fillna(0)
        maxdiff = diff.max()
        i+=1
        if i>3000:
            maxdiff = 0
    print(f'Vx clean iter:{i}')
    ######### Vy
    diff = abs(plasma['vy'].diff())
    diff.fillna(0)
    maxdiff = diff.max()
    i=0
    while maxdiff>100:
        plasma = plasma[diff<100]
        diff = abs(plasma['vy'].diff())
        diff.fillna(0)
        maxdiff = diff.max()
        i+=1
        if i>3000:
            maxdiff = 0
    print(f'Vy clean iter:{i}')
    ######### Vz
    # Seemed good by here
    #for key in ['n','p','vx','vy','vz']:
    for key in ['ne','p','vx','vy','vz']:
        cleaned_dict['plasma'][key] = plasma[key]
    cleaned_dict['loc'] = probedata['loc']
    cleaned_dict['imf'] = probedata['imf']
    return cleaned_dict

def shift_to_upstream(probedata):
    shifted_data = pd.DataFrame()
    shift_dict = {}
    e = 1.602e-19 #electron charge
    kb = 1.380e-23 #J/K boltzmann constant
    loctimes =(probedata['loc'].index-dt.datetime(2024,5,9,0)).total_seconds()
    imftimes =(probedata['imf'].index-dt.datetime(2024,5,9,0)).total_seconds()
    plasmatimes = (probedata['plasma'].index-
                                      dt.datetime(2024,5,9,0)).total_seconds()
    #NOTE RBE doesnt read msec so needspacing of at least 1s to not confuse it
    #tstart = int(plasmatimes[0])
    #tend = int(plasmatimes[-1])
    #nsec = tend-tstart+1
    #eq_times = np.linspace(tstart,tend,nsec)
    xNose = 32 #Where we put the solar wind into the simulation
    '''
    X =(np.interp(eq_times,loctimes,probedata['loc']['x_gse'])-xNose)*6371
    Y = np.interp(eq_times,loctimes,probedata['loc']['y_gse'])*6371
    N = np.interp(eq_times,plasmatimes,probedata['plasma']['ne'])
    P = np.interp(eq_times,plasmatimes,probedata['plasma']['p'])
    Vx = np.interp(eq_times,plasmatimes,probedata['plasma']['vx'])
    Vy = np.interp(eq_times,plasmatimes,probedata['plasma']['vy'])
    Vz = np.interp(eq_times,plasmatimes,probedata['plasma']['vz'])
    Bx = np.interp(eq_times,imftimes,probedata['imf']['bx'])
    By = np.interp(eq_times,imftimes,probedata['imf']['by'])
    Bz = np.interp(eq_times,imftimes,probedata['imf']['bz'])
    '''
    X = (probedata['loc']['x_gse']-xNose)*6371
    Y = probedata['loc']['y_gse']*6371
    #N = np.interp(loctimes,plasmatimes,probedata['plasma']['n'])
    N = np.interp(loctimes,plasmatimes,probedata['plasma']['ne'])
    P = np.interp(loctimes,plasmatimes,probedata['plasma']['p'])
    Vx = np.interp(loctimes,plasmatimes,probedata['plasma']['vx'])
    Vy = np.interp(loctimes,plasmatimes,probedata['plasma']['vy'])
    Vz = np.interp(loctimes,plasmatimes,probedata['plasma']['vz'])
    Bx = np.interp(loctimes,imftimes,probedata['imf']['bx'])
    By = np.interp(loctimes,imftimes,probedata['imf']['by'])
    Bz = np.interp(loctimes,imftimes,probedata['imf']['bz'])

    shifted_data['vx'] = Vx
    shifted_data['vy'] = Vy
    shifted_data['vz'] = Vz
    shifted_data['bx'] = Bx
    shifted_data['by'] = By
    shifted_data['bz'] = Bz
    shifted_data['density'] = N
    shifted_data['temperature'] = P*e/(N*kb)
    # Shift formula from:
    #   https://omniweb.gsfc.nasa.gov/html/ow_data.html#time_shift
    #V = np.sqrt(probedata['plasma']['vx']**2+probedata['plasma']['vy']**2+
    #            probedata['plasma']['vz']**2).rolling('3600s').mean()
    V = np.sqrt(Vx**2+Vy**2+Vz**2)
    W = np.tan(0.5*np.arctan2(V,428))
    Ve = 30
    tshift = (X/V) * ((1 + (Y*W)/X)/(1 - Ve*W/V))

    # Repack into a shifted time
    shifted_data['times'] = [dt.datetime(2024,5,9,0)+dt.timedelta(seconds=t)
                                           for t in loctimes+tshift.values]
    #                                       for t in eq_times+tshift]
    shifted_data = shifted_data.sort_values(by='times')
    '''
    shifted_data.index = shifted_data['times']
    tshifted = (shifted_data.index-dt.datetime(2024,5,9,0)).total_seconds()
    tstart = int(tshifted[0])
    tend = int(tshifted[-1])
    nsec = tend-tstart+1
    eq_times2 = np.linspace(tstart,tend,nsec)
    shift_dict['vx'] = np.interp(eq_times2,tshifted,Vx)
    shift_dict['vy'] = np.interp(eq_times2,tshifted,Vy)
    shift_dict['vz'] = np.interp(eq_times2,tshifted,Vz)
    shift_dict['bx'] = np.interp(eq_times2,tshifted,Bx)
    shift_dict['by'] = np.interp(eq_times2,tshifted,By)
    shift_dict['bz'] = np.interp(eq_times2,tshifted,Bz)
    shift_dict['density'] = np.interp(eq_times2,tshifted,N)
    shift_dict['temperature'] = np.interp(eq_times2,tshifted,P*e/(N*kb))
    shift_dict['times'] = [dt.datetime(2024,5,9,0)+dt.timedelta(seconds=t)
                                                            for t in eq_times2]
    '''
    '''
    # Additional Quality correct the data
    # avoid small sets of isolated points
    island_size = 3 #NOTE only tested for n=3
    #get a list of all gaps
    tdelta = [t.total_seconds() for t in shifted_data['times'].diff()]
    tdelta[0] = tdelta[1]
    gaps = np.zeros([len(tdelta),2*(island_size)])
    #arrange all gaps touching n consecutive points where n is island size
    k=0
    for i in range(1,island_size+1):
        if i==1:
            back = np.array(tdelta)
            fwd = np.append(tdelta[1::],[60])
        else:
            back = np.append((i-1)*[60],tdelta[:-(i-1)])
            fwd = np.append(tdelta[i::],i*[60])
        gaps[:,k] = back
        gaps[:,k+1] = fwd
        k+=2
    # if the second largest gap is > the 99.9th percentile we have an island
    second_largest_gap = [g[np.argsort(g)[-2]] for g in gaps]
    isIsland = second_largest_gap>=np.float64(120.)
    shifted_data = shifted_data[~isIsland]
    '''

    # only keep times around the event window
    for key in shifted_data.keys():
        if 'time' not in key:
            shift_dict[key] = shifted_data[key].values
    shift_dict['times'] = shifted_data['times']

    return shift_dict

#Main program
if __name__ == '__main__':
    interval = [dt.datetime(2024,5,9,0),dt.datetime(2024,5,15,0)]
    # Get data from CDAweb if we havent done it yet
    if not os.path.exists('themis_pos.h5'):
        th_pos,th_bfield,th_plasma = collect_themis(interval[0],interval[1],
                                                    probelist=['B','C'])
    # Read in data
    themisB, themisC = {},{}
    loc = pd.HDFStore('themis_pos.h5')
    imf = pd.HDFStore('themis_bfield.h5')
    plasma = pd.HDFStore('themis_plasma.h5')
    themisB['loc'] = loc['/themisB'].dropna()
    themisB['imf'] = imf['/themisB'].dropna()
    themisB['plasma'] = plasma['/themisB'].dropna()
    loc.close()
    imf.close()
    plasma.close()

    # Clean data
    cleaned_data = clean_data(themisB)

    # Shift data to upsteam location
    shifted_data = shift_to_upstream(cleaned_data)

    # Convert shifted data into IMF.dat input file
    #imffile = convert_to_IMF(shifted_data)
    write_imf_input(shifted_data,coords='GSM')
