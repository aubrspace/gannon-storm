#!/usr/bin/env python3
""" Extract DMSP data and SWMF-IE data together into a single source for plots
"""
import os,sys,glob,time
import numpy as np
import datetime as dt
from tqdm import tqdm

#############################################################################
def combine_ie(indata:list[dict:np.ndarray]) -> dict[str:np.ndarray]:
    outdata = {}
    for key in indata[0]:
        outdata[key] = np.array([d[key] for d in indata])
    return outdata

def load_ie(infile:str) -> dict[str:np.ndarray]:
    data = {}
    with open(infile,'r') as f:
        # Read header/aux info
        for line in f:
            if 'TITLE' in line:
                # Handle this later
                titleline = line
            elif 'VARIABLE' in line:
                # variables are split by commas & newlines with "" around each
                variablelines = line
                done_w_variables = False
                i = 0
                while not done_w_variables:
                    line = f.readline()
                    i+=1
                    if 'ZONE' in line:
                        zoneline = line
                        zone=zoneline.split('T=')[1].split()[0].replace('"','')
                        data[zone] = {}
                        done_w_variables = True
                    elif i>50:
                        done_w_variables = True
                    else:
                        variablelines += line
                variables = [v for v in variablelines.split('"') if '[' in v]
            if 'ZONE' in line:
                # Zone gives us the name
                zoneline = line
                zone = zoneline.split('T=')[1].split()[0].replace('"','')
                data[zone] = {}
            if 'I=' in line and 'J=' in line:
                # Grid is in lat (i) and lon (j)
                gridline = line
                data[zone]['grids'] = np.array([int(v) for v in
                                           np.array(gridline.split())[[1,3]]])
                nlines = data[zone]['grids'][0] * data[zone]['grids'][1]
                # initialize arrays
                for variable in variables:
                    data[zone][variable] = np.zeros(nlines)
                # we now know how much data should be in this block
                for i in range(0,nlines):
                    raw1 = f.readline()
                    raw2 = f.readline()
                    rawdata = np.concat([raw1.split(),raw2.split()])
                    for i,variable in enumerate(variables):
                        data[zone][variable][i] = float(rawdata[i])
    # take some helpful aux data from the title line earlier
    time_str, btilt_str = np.array(titleline.split())[[4,6]] #NOTE fragile
    for zone in data:
        data[zone]['time'] = dt.datetime.strptime(time_str,
                                                     "%Y-%m-%d-%H-%M-%S-000,")
        data[zone]['btilt'] = np.array(float(btilt_str))
    return data

def read_all_ie(inpath:str) -> dict[dict[str:np.ndarray]]:
    ie = {}
    print(f"Loading files from {inpath}")
    files = glob.glob(f"{inpath}/*.tec")
    results_N = [[]]*len(files)
    results_S = [[]]*len(files)
    for i,f in enumerate(tqdm(files)):
        result = load_ie(f)
        results_N[i] = result['IonN']
        results_S[i] = result['IonS']
    ie['N'] = combine_ie(results_N)
    ie['S'] = combine_ie(results_S)
    return ie

#############################################################################
def combine_dmsp_passes(indata:list[dict[str:np.ndarray]]
                                                    ) -> dict[str:np.ndarray]:
    """Combines multiple dictionaries of dmsp data into one
    """
    outdata = {}
    for key in indata[0]:
        outdata[key] = np.concat([col[key] for col in indata])
    order = np.argsort(outdata['time'])
    for key in outdata:
        outdata[key] = outdata[key][order]
    return outdata

def load_dmsp(infile:str) -> np.ndarray:
    data = {}
    # Read data as comma separated
    with open(infile,'r') as f:
        headers = f.readline().replace('\n','').split(',')
        rawdata = f.readlines()
    # dice up the elements
    rawdata_split = np.array([line.replace('\n','').split(',')
                                                         for line in rawdata])
    # match up columns with headers
    for i,variable in enumerate(headers):
        data[variable] = rawdata_split[:,i]
    # clean up dictionary by adjusting the data types
    time_strs = data.pop('datetime')
    for key in [k for k in data if '_id' not in k]:
        data[key][data[key]==''] = 'Nan'
        data[key] = np.array([float(v) for v in data[key]])
    data['time']=np.array([dt.datetime.fromisoformat(t) for t in time_strs])
    return data

def read_all_dmsp(inpath:str) -> dict[dict[str:np.ndarray]]:
    dmsp = {}
    paths = [f for f in glob.glob(f"{inpath}/*") if 'compiled' not in f]
    for path in paths:
        print(f"Loading files from {path} ...")
        folder = path.split('/')[-1]
        tag = folder.split('passes_')[-1]
        files = glob.glob(f"{inpath}/{folder}/*.csv")
        results = [[]]*len(files)
        for i,f in enumerate(tqdm(files)):
            results[i] = load_dmsp(f)
        dmsp[tag+'_N'] = combine_dmsp_passes(
                               [r for r in results if 'N' in r['pass_id'][0]])
        dmsp[tag+'_S'] = combine_dmsp_passes(
                               [r for r in results if 'S' in r['pass_id'][0]])
    return dmsp

#############################################################################

def main() -> None:
    dmsppath = "../data/dmsp"
    iepath = "../data/large/IE/ionosphere"
    if False:
        # Read in DMSP data
        dmsp = read_all_dmsp(dmsppath)
        # Write out to npz file
        for tag in dmsp:
            outfile = f"{dmsppath}/compiled_{tag}.npz"
            np.savez_compressed(outfile,**dmsp[tag])
            print(f'\033[92m Created\033[00m {outfile}')
        print('\n\n')
    # Read in IE data
    if True:
        ie = read_all_ie(iepath)
        # Write out to npz file
        for hemi in ie:
            outfile = f"{iepath}/compiled_{hemi}.npz"
            np.savez_compressed(outfile,**ie[hemi])
            print(f'\033[92m Created\033[00m {outfile}')
    return

if __name__ == "__main__":
    main()
