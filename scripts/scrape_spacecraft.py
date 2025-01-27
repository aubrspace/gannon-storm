#/usr/bin/env python
"""accesses WIND data from nasa CDA and creates IMF_new.dat for SWMF input
"""
import os,sys
sys.path.append(os.getcwd().split('swmf-energetics')[0]+
                                      'swmf-energetics/')
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
# Custom
from global_energetics.wind_to_swmfInput import (collect_themis,collect_mms,
                                             collect_cluster,collect_geotail,
                                             collect_goes)

def write_SWMF_satfile(df,outname,outpath,**kwargs):
    """ function writes data from DataFrame to swmf satfile
    Inputs
    Returns
    """
    with open(os.path.join(outpath,outname),'w') as f:
        if 'note' in kwargs:
            f.write(kwargs.get('note')+'\n')
        f.write('Year Mo Dy Hr Mn Sc Msc X Y Z\n')
        f.write('\n')
        f.write('#COORD\n')
        f.write(kwargs.get('coord','GSM\n'))
        f.write('\n')
        f.write('#START\n')
        for t,(x,y,z) in zip(df.index,df[['x_gsm','y_gsm','z_gsm']].values):
            line =[]
            line.append(f'{t.year}')
            line.append(f'{t.month}')
            line.append(f'{t.day}')
            line.append(f'{t.hour:02n}')
            line.append(f'{t.minute:02n}')
            line.append(f'{t.second:02n}')
            line.append(f'{t.microsecond*1e3:03n}')
            line.append(f'{x:.2f}')
            line.append(f'{y:.2f}')
            line.append(f'{z:.2f}')
            f.write(''.join([s.rjust(7) for s in line])[3::]+'\n')
        print(f'\tCreated {os.path.join(outpath,outname)}')

#Main program
if __name__ == '__main__':
    #############################USER INPUTS HERE##########################
    start = dt.datetime(2024,5,10,6)
    end = dt.datetime(2024,5,13,0)
    outpath = './gannon_storm/'
    plot_data = False
    #######################################################################

    # Scrape data from CDAWeb
    goes_pos, goes_b, goes_plasma = collect_goes(start,end,
                                            skip_bfield=True,skip_plasma=True,
                                            probes=['16'],
                                            writeData=True)
    cluster_pos, cluster_b,cluster_plasma = collect_cluster(start, end,
                                            skip_bfield=True,skip_plasma=True,
                                            probes=['1','2','3','4'],
                                            writeData=True)
    mms_pos, mms_b,mms_plasma = collect_mms(start, end,
                                            skip_bfield=True,skip_plasma=True,
                                            probes=['1','2','3','4'],
                                            writeData=True)
    themis_pos, themis_b,themis_plasma = collect_themis(start, end,
                                            skip_bfield=True,skip_plasma=True,
                                            probes=['A','B','C','D','E'],
                                            writeData=True)

    # Quick Plot
    quicklook,(equitorial,meridional) =plt.subplots(1,2,figsize=[20,10])
    for sat in cluster_pos.keys():
        equitorial.scatter(cluster_pos[sat]['x_gsm'],cluster_pos[sat]['y_gsm'],
                    label=sat)
        meridional.scatter(cluster_pos[sat]['x_gsm'],cluster_pos[sat]['z_gsm'],
                    label=sat)
    for sat in themis_pos.keys():
        equitorial.scatter(themis_pos[sat]['x_gsm'],themis_pos[sat]['y_gsm'],
                    label=sat)
        meridional.scatter(themis_pos[sat]['x_gsm'],themis_pos[sat]['z_gsm'],
                    label=sat)
    for sat in mms_pos.keys():
        equitorial.scatter(mms_pos[sat]['x_gsm'],mms_pos[sat]['y_gsm'],
                           label=sat)
        meridional.scatter(mms_pos[sat]['x_gsm'],mms_pos[sat]['z_gsm'],
                           label=sat)
    for sat in goes_pos.keys():
        equitorial.scatter(goes_pos[sat]['x_gsm'],goes_pos[sat]['y_gsm'],
                           label=sat)
        meridional.scatter(goes_pos[sat]['x_gsm'],goes_pos[sat]['z_gsm'],
                           label=sat)
    equitorial.set_xlim(20,-20)
    equitorial.set_ylim(20,-20)
    equitorial.set_xlabel('X [R]')
    equitorial.set_ylabel('Y [R]')
    equitorial.legend()
    meridional.set_xlim(20,-20)
    meridional.set_ylim(-20,20)
    meridional.set_xlabel('X [R]')
    meridional.set_ylabel('Z [R]')
    meridional.legend()
    from IPython import embed; embed()

    print('Writing Cluster Satfiles ...')
    for sat in cluster_pos.keys():
        write_SWMF_satfile(cluster_pos[sat],sat+'.txt','./gannon_storm',
                           note=f'{sat} Created {str(dt.datetime.now())}')
    print('Writing MMS Satfiles ...')
    for sat in mms_pos.keys():
        write_SWMF_satfile(mms_pos[sat],sat+'.txt','./gannon_storm',
                           note=f'{sat} Created {str(dt.datetime.now())}')
    print('Writing THEMIS Satfiles ...')
    for sat in themis_pos.keys():
        write_SWMF_satfile(themis_pos[sat],sat+'.txt','./gannon_storm',
                           note=f'{sat} Created {str(dt.datetime.now())}')
    print('Writing GOES Satfiles ...')
    for sat in goes_pos.keys():
        write_SWMF_satfile(goes_pos[sat],sat+'.txt','./gannon_storm',
                           note=f'{sat} Created {str(dt.datetime.now())}')
