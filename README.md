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

The provided datasets within the ```finance_data``` folder are named in the form date_numDays_lbSize_correlationType_useOfSelfLoops. For this data we do not manually add self loops at any point as the GCNConv function does this automatically. These correspond to the 500 days of train/test data. They are all in the form of a (zipped) PyTorch Geometric Temporal "DynamicGraphTemporalSignal" object.

The ```GDN_outputs.py``` and ```GCN_outputs.py``` folders contain the respective model outputs, $S_t^\alpha(\theta)$, for each of the datasets. The files name format is model_numDays_featureSize_GDNNumEpochs_DDPMNumEpochs_gamma_delta_GDNLearningRate_DDPMLearningRate_DDPMlb_traininglb_dataName. The ```GDN_weights.py``` and ```GCN_weights.py``` folders contain the equivalent files but now for model weights at the end of the respective 100 day train/tets period.

Finally, the ```analysis.ipynb``` notebook provides the code to reproduce Figure 4, Figure 5, the Sharpe Ratio tests and the t-tests.
