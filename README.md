# Repository for the paper "Graph Denoising Networks: A Deep Learning Framework for Equity Portfolio Construction"



## Requirements
```
Python 3.9
torch
torch_geometric
```
## Overview
The ```model.py``` and ```train_test.py``` files provide the raw code needed to create and then train/test the GDN model.

The ```example.ipynb``` notebook provides a simple notebook that shows an implimentation of the GDN we use.

The provided datasets (YET TO BE UPLOADED) within the ```finance_data``` folder are in the form date_numDays_lbSize_correlationType_useOfSelfLoops. For this implementation we do not manually add self loops at any point as the GCNConv function does this automatically.
