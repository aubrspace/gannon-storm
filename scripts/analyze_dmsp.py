#!/usr/bin/env python3
"""Analysis and plots for the Gannon storm GRL paper comparisons with DMSP
"""
import os,sys,glob,time
import numpy as np
from matplotlib import pyplot as plt

def main() -> None:
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

    from IPython import embed; embed()
    # For each satellite
    #   Plot just the crossings in N hemi SM coordinated w/ Ex color
    #   Downsample to just the crossing times
    #   From each crosing
    #       extract IE along the crossing
    #       integrate E along each crossing
    #       adjustment for altitude???
    return

if __name__ == "__main__":
    main()
