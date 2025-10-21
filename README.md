# This repo contains input files, analysis data, and plotting scripts for the journal article "May 10, 2024 Gannon Storm: Connecting Energy Flux with Field Aligned Currents and Cross Polar Cap Potential"
### Contents
- LICENSE
- README.md
- **data**: files containing observation/simulationo data
  - **ampere**: AMPERE network derived field-aligned currents files
  - **analysis**: SWMF simulation analysis output data
  - **dmsp**: DMSP satellite derived E field and cross polar cap potential
  - **logs**: SWMF simulation logs
  - **pc_index**: polar cap index
  - **sat**: satellite data files (real and virtual)
- **inputs**: files used to run the SWMF simulation
- **outputs**: figures or visualizations produced for the paper
- **scripts**: python code used to produce the outputs

# Notes for file formats
- **.png**: image files
- **.pdf**: pdf exported image files
- **.csv**: human readable ASCII
- **.log**: human readable ASCII
- **.dat**: human readable ASCII
- **.sat**: human readable ASCII
- **.in**: human readable ASCII
- **.txt**: human readable ASCII
- **.h5**: HDF5 format, can be read with python pandas module.
- **.npz**: numpy format, compressed set of numpy arrays. See numpy documentation for details
- **.py**: python script
