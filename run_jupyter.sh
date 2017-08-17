#!/bin/bash
# calling jupyter with this option avoids an error message regarding bandwidth rescrictions leading to errors execution the notebooks for the PNAS paper
jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000 
